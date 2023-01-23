# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from multiprocessing import reduction
from statistics import mode

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass, field

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@dataclass
class CrossEntropyWithRobustAllConfig(
    LabelSmoothedCrossEntropyCriterionConfig):

    reg_alpha: float = field(
       default=1.5, metadata={"help":""}
    )
    kl_direction: str = field(
        default='both', metadata={"help": "both clean noise"}
    )
    only_nll: str = field(
        default='both', metadata={"help": "both clean noise"}
    )
  
@register_criterion(
    "cross_entropy_with_robust_all",
    dataclass=CrossEntropyWithRobustAllConfig,
)

class CrossEntropyWithRobustAll(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, reg_alpha, kl_direction, only_nll):
        super().__init__(task, sentence_avg, label_smoothing)
        
        self.task = task

        self.reg_alpha = reg_alpha        
        self.kl_direction = kl_direction
        self.only_nll = only_nll

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        probs = model.get_normalized_probs(net_output, log_probs=False)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            probs = probs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), probs.view(-1, probs.size(-1)), target.view(-1)

    def forward(self, model, sample, reduce=True):
       # print(self.only_one_side_kl) 
        sample_input = sample['net_input']
        sample_concat_input = {
            'src_tokens': torch.cat([sample_input['src_tokens'], sample_input['src_tokens'].clone()], 0),
            'src_lengths': torch.cat([sample_input['src_lengths'], sample_input['src_lengths'].clone()], 0),
            'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
        }
        
        net_output = model(**sample_concat_input)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        #print('lprobs=>')
        target = model.get_targets(sample, net_output)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        target = torch.cat([target, target.clone()], dim=0)
        #print('target.size()',target.size())
        #print('lprobs.size()',lprobs.size())
        if self.only_nll == 'both':
           p,t = lprobs,target
           #pm = pad_mask
        elif self.only_nll == 'clean':
           p,t = lprobs[-lprobs.size(0)//2:,:],target[-target.size(0)//2:]
           #print('p.size()',p.size())
           #print('t.size()',t.size())
           #pm = t.unsqueeze(-1).eq(self.padding_idx)
        elif self.only_nll == 'noise':
           p,t = lprobs[:lprobs.size(0)//2,:],target[:target.size(0)//2]
           #pm =  t.unsqueeze(-1).eq(self.padding_idx)
        loss, nll_loss = label_smoothed_nll_loss(
            p, t.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        ntokens = sample['ntokens']
        nsentences = sample['target'].size(0)
        sample_size = sample['ntokens']

        #if self.extra_loss_type == 'kl':
        kl_loss = self.compute_kl_loss(model, net_output, pad_mask,reduce,self.kl_direction)
        loss += self.reg_alpha * kl_loss
            
        #elif self.extra_loss_type == 'c' and self.training:
        #    c_loss = reg_alpha * self.get_contrastive_loss(net_output[0],pad_mask) * ntokens/nsentences
        #    # print('using cl loss')
        #    loss += reg_alpha * c_loss

        #if ignore_grad:
        #    loss *= 0
        #with torch.autograd.profiler.record_function("backward"):
        #    v(loss)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
             "kl_loss": utils.item(kl_loss.data) if reduce else kl_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True,kl_direction='both'):
        # print(only_one_side_kl)
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
        #print('q.size():',q.size())        

        if kl_direction == 'clean':
           p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
           #q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        elif kl_direction == 'noise':
           #print('cal noise kl loss')
           q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
           #p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        else: 
           p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
           q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        
        if pad_mask is not None:
            if kl_direction == 'clean':
               p_loss.masked_fill_(pad_mask, 0.)
               #q_loss.masked_fill_(pad_mask, 0.)
            elif kl_direction == 'noise':
               q_loss.masked_fill_(pad_mask, 0.)
               #p_loss.masked_fill_(pad_mask, 0.) 
               #print('pad noise kl')
            else:
               p_loss.masked_fill_(pad_mask, 0.)
               q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            if kl_direction == 'clean':
               p_loss = p_loss.sum()
               # q_loss = q_loss.sum()
            elif kl_direction == 'noise':
               q_loss = q_loss.sum() 
               #print('sum noise kl')
            else:
               p_loss = p_loss.sum()
               q_loss = q_loss.sum()
        
        if kl_direction == 'clean':
           loss = p_loss
        elif kl_direction == 'noise':
           loss = q_loss
        else:
           loss = (p_loss + q_loss) / 2
        return loss

    def compute_loss(self, model, net_output_ln_kept, net_output_ln_removed, sample, reduce=True, addition=None):

        lprobs_ln_removed, probs_ln_removed, target = self.get_lprobs_and_target(model, net_output_ln_removed, sample)
        lprobs_ln_kept, probs_ln_kept, target = self.get_lprobs_and_target(model, net_output_ln_kept, sample)

        ce_loss_ln_removed, nll_loss_ln_removed = label_smoothed_nll_loss(
            lprobs_ln_removed,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        ce_loss_ln_kept = None 
        if model.num_updates % 1:
            ce_loss_ln_kept, _ = label_smoothed_nll_loss(
                lprobs_ln_kept,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
        
        kl_loss = self.compute_ln_reg_loss(probs_ln_kept, probs_ln_removed, target, reduce=reduce)
        
        return ce_loss_ln_removed, ce_loss_ln_kept, kl_loss,  nll_loss_ln_removed

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )

        kl_loss_sum = utils.item(sum(log.get("kl_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
 
        metrics.log_scalar("kl_loss", kl_loss_sum)
        metrics.log_scalar('ntokens',ntokens) 
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
