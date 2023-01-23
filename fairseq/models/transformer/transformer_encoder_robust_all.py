# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
import json 
import random
# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        
        self.cfg = cfg
        self.cur_max_noise_rate = 0.0
        self.src_dict = dictionary

        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        layer_norm_dropout_rate = -1.0, #(1.0 if (not self.training and self.cfg.layer_norm_dropout != -1) else self.cfg.layer_norm_dropout),
        num_updates=0,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        src_tokens,src_lengths,cur_max_noise_rate = replace_tokens(src_tokens,src_lengths,cur_max_noise_rate=self.cur_max_noise_rate,src_dict=self.src_dict,training_state=self.training,num_updates=num_updates,args=self.cfg)

        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, layer_norm_dropout_rate=layer_norm_dropout_rate
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        layer_norm_dropout_rate=-1,
        
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # nested tensor and BT enable
        layer = self.layers[0]
        BT_flag = False
        NT_flag = False
        # torch version check, BT>=1.12.0 and NT>=1.13.0.dev20220613
        # internal format is '1.13.0a0+fb'
        # external format is '1.13.0.dev20220613'(cpu&gpu) for nightly or "1.11.0"(cpu) or '1.11.0+cu102'(gpu) for stable
        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                int(torch_version[0]) * 1000
                + int(torch_version[1]) * 10
                + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True
                # Consider _nested_tensor_from_mask_left_aligned is landed after "20220613"
                if int_version >= 1131 or (
                    int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
            BT_version
            and x.dim() == 3
            and layer.load_to_BT
            and not layer.return_fc
            and layer.can_use_fastpath
            and not layer.training
            and not layer.ever_training
            and not layer.cfg_checkpoint_activations
        ):
            # Batch first can not be justified but needs user to make sure
            x = x.transpose(0, 1)
            # Check mask conditions for nested tensor
            if NT_version:
                if (
                    encoder_padding_mask is not None
                    and torch._nested_tensor_from_mask_left_aligned(
                        x, encoder_padding_mask.logical_not()
                    )
                ):
                    if not torch.is_grad_enabled() or not x.requires_grad:
                        x = torch._nested_tensor_from_mask(
                            x, encoder_padding_mask.logical_not()
                        )
                        NT_flag = True
            BT_flag = True

        # encoder layers
        if NT_flag:
            processing_mask = None
        else:
            processing_mask = encoder_padding_mask
        encoder_padding_mask_out = processing_mask if has_pads else None
        for layer in self.layers:
            lr = layer(x, encoder_padding_mask=encoder_padding_mask_out, layer_norm_dropout_rate=layer_norm_dropout_rate)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        # change back to non-nested and Batch second
        if NT_flag:
            x = x.to_padded_tensor(0.0)

        if NT_flag or BT_flag:
            x = x.transpose(0, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

def replace_tokens(src_tokens,src_lengths,cur_max_noise_rate,src_dict,training_state,num_updates,args):

    def cl_noise_rate(num_updates, max_num_updates, R_max, R_min=0.0, p=2, reverse=0):
        square_Rmax = R_max ** p
        square_Rmin = R_min ** p if R_min != 0 else 0

        temp = ((square_Rmax - square_Rmin) * (num_updates / max_num_updates) + square_Rmin)
        if reverse:
            #print('decrease')
            return max(R_max - temp ** (1 / p) if temp != 0 else R_max, 0)
        else:
            #print('increase')
            return min(temp ** (1 / p) if temp != 0 else 0, R_max)

    add_noise = getattr(args,'add_noise',False)
    noise_rate = getattr(args,'noise_rate',None)
    is_half_batch = getattr(args,'is_half_batch',False)
    clearning = getattr(args,'curriculum_learning',False)
    cargs =  getattr(args,'curriculum_args',None)
    cargs = json.loads(args.curriculum_args) if cargs is not None else None
    noise_type = getattr(args,'noise_type','replace')
    if noise_type == 'insert':
        src_tokens_extend = torch.zeros(size=(src_tokens.size(0),src_tokens.size(1)+1)).to(src_tokens)
    #print(noise_rate)
    if training_state and add_noise and (noise_rate !=0):
        bsz = src_tokens.size(0)//2 if is_half_batch else src_tokens.size(0)
        for i,s in enumerate(src_tokens[:bsz]): # each sentence
            if clearning and cargs:
                max_rate = cargs['max_rate']
                min_rate = cargs['min_rate']
                cupdates =  cargs['cupdates']
                mupdates = cargs['mupdates']
                reverse = cargs['reverse'] if 'reverse' in cargs else 0
                p = cargs['p']
                cl_type = cargs['type'] if 'type' in cargs else 1

                if num_updates % cupdates == 0:
                    cur_max_noise_rate = cl_noise_rate(
                        num_updates=num_updates,
                        R_max=max_rate,
                        R_min=min_rate,
                        p=p,
                        max_num_updates=mupdates,
                        reverse=reverse,
                    )
                #print(cur_max_noise_rate)
                if cl_type == 2:
                    noise_rate = cl_noise_rate(
                        num_updates=num_updates,
                        R_max=max_rate,
                        R_min=min_rate,
                        p=p,
                        max_num_updates=mupdates,
                        reverse=reverse,
                    )
                elif cl_type == 3:
                    #print('sample from fixed range')
                    noise_rate = random.uniform(0, max_rate)
                else:
                    noise_rate = random.uniform(0, cur_max_noise_rate)
            if noise_rate == 0.0:
               break
            start_idx = 0
            end_idx = len(s) - 1
            if s[0].cpu().numpy() != src_dict.pad(): # 无填充
                # 采样需要替换的位置
                n_sampled = math.ceil(noise_rate * end_idx)

                if noise_type == 'replace':
                    idx_list1 = torch.LongTensor(random.sample(range(start_idx,end_idx),n_sampled)).to(src_tokens)
                    idx_list2 = torch.LongTensor(random.sample(range(4,len(src_dict.symbols)),n_sampled)).to(src_tokens)
                    s_rp = s.scatter(0,idx_list1,idx_list2)
                    src_tokens[i] = s_rp
                elif noise_type == 'remove':
                    idx_reserved = random.sample(range(start_idx,end_idx),end_idx - n_sampled)+[end_idx]
                    idx_reserved.sort()
                    idx_reserved = torch.LongTensor(idx_reserved).to(src_tokens)
                    pad_tensor = torch.LongTensor([src_dict.pad()] * n_sampled).to(src_tokens)
                    src_tokens[i] = torch.cat((pad_tensor,s[idx_reserved]),0).to(src_tokens)
                    src_lengths[i] = src_lengths[i] - n_sampled
                elif noise_type == 'swap':
                    if start_idx >= end_idx-1 or n_sampled > end_idx-1-start_idx:
                       continue
                    origial_src = s.clone()
                    idx_swap1 = torch.LongTensor(random.sample(range(start_idx,end_idx-1),n_sampled)).to(src_tokens)
                    idx_swap2 = idx_swap1 + 1
                    s[idx_swap1] = origial_src[idx_swap2]
                    s[idx_swap2] = origial_src[idx_swap1]
                    src_tokens[i] = s
                elif noise_type == 'hybrid':
                    rd = random.uniform(0,1)
                    rd1,rd2 = 1/3,2/3
                    if rd >= 0 and rd < rd1: # replace
                        # replace
                        idx_list1 = torch.LongTensor(random.sample(range(start_idx,end_idx),n_sampled)).to(src_tokens)
                        idx_list2 = torch.LongTensor(random.sample(range(4,len(src_dict.symbols)),n_sampled)).to(src_tokens)
                        s_rp = s.scatter(0,idx_list1,idx_list2)
                        src_tokens[i] = s_rp
                    elif rd >= rd1 and rd < rd2: # remove
                        # remove
                        idx_reserved = random.sample(range(start_idx,end_idx),end_idx - n_sampled)+[end_idx]
                        idx_reserved.sort()
                        idx_reserved = torch.LongTensor(idx_reserved).to(src_tokens)
                        pad_tensor = torch.LongTensor([src_dict.pad()] * n_sampled).to(src_tokens)
                        src_tokens[i] = torch.cat((pad_tensor,s[idx_reserved]),0).to(src_tokens)
                        src_lengths[i] = src_lengths[i] - n_sampled
                    elif rd >= rd2 and rd < 1: # swap
                        # swap
                        if start_idx >= end_idx-1 or n_sampled > end_idx-1-start_idx:
                           continue
                        origial_src = s.clone()
                        idx_swap1 = torch.LongTensor(random.sample(range(start_idx,end_idx-1),n_sampled)).to(src_tokens)
                        idx_swap2 = idx_swap1 + 1
                        s[idx_swap1] = origial_src[idx_swap2]
                        s[idx_swap2] = origial_src[idx_swap1]
                        src_tokens[i] = s
                elif noise_type == 'insert':

                    idx_list1 = random.sample(range(start_idx,end_idx),n_sampled)
                    idx_list1.sort()
                    idx_list2 = random.sample(range(4,len(src_dict.symbols)),n_sampled)
                    s_list = s.tolist()

                    #print('insert => ',idx_list1)
                    #print('before => ',src_tokens[i])
                    #print('length => ',src_lengths[i])

                    for insert_id,insert_idx in enumerate(idx_list1):
                        s_list.insert(insert_idx,idx_list2[insert_id])

                    src_tokens_extend[i] = torch.LongTensor(s_list).to(src_tokens)
                    src_lengths[i] = src_lengths[i] + n_sampled
            else: # 填充
                # 确定填充数量
                start_idx=1
                for idx,word_idx in enumerate(s[1:],start=1):
                    if word_idx.cpu().numpy() != src_dict.pad():
                        start_idx = idx
                        break
                n_sampled = math.ceil(noise_rate * (end_idx-start_idx))
                if noise_type == 'replace':
                    idx_list1 = torch.LongTensor(random.sample(range(start_idx,end_idx),n_sampled)).to(src_tokens)
                    idx_list2 = torch.LongTensor(random.sample(range(4,len(src_dict.symbols)),n_sampled)).to(src_tokens)
                    s_rp = s.scatter(0,idx_list1,idx_list2)
                    src_tokens[i] = s_rp
                elif noise_type == 'remove':
                    pad_tensor = torch.LongTensor([src_dict.pad()] * (n_sampled + start_idx)).to(src_tokens)
                    idx_reserved = random.sample(range(start_idx,end_idx),end_idx - n_sampled-start_idx)+[end_idx]
                    idx_reserved.sort()
                    idx_reserved = torch.LongTensor(idx_reserved).to(src_tokens)
                    src_tokens[i] = torch.cat((pad_tensor,s[idx_reserved]),0).to(src_tokens)
                    src_lengths[i] = src_lengths[i] - n_sampled
                elif noise_type == 'swap':
                    if start_idx >= end_idx-1 or n_sampled > end_idx-1-start_idx:
                       continue
                    origial_src = s.clone()
                    idx_swap1 = torch.LongTensor(random.sample(range(start_idx,end_idx-1),n_sampled)).to(src_tokens)
                    idx_swap2 = idx_swap1 + 1
                    s[idx_swap1] = origial_src[idx_swap2]
                    s[idx_swap2] = origial_src[idx_swap1]
                    src_tokens[i] = s
                elif noise_type == 'hybrid':
                    rd = random.uniform(0,1)
                    rd1,rd2 = 1/3,2/3
                    if rd >= 0 and rd < rd1: # replace
                        # replace
                        idx_list1 = torch.LongTensor(random.sample(range(start_idx,end_idx),n_sampled)).to(src_tokens)
                        idx_list2 = torch.LongTensor(random.sample(range(4,len(src_dict.symbols)),n_sampled)).to(src_tokens)
                        s_rp = s.scatter(0,idx_list1,idx_list2)
                        src_tokens[i] = s_rp
                    elif rd >= rd1 and rd < rd2: # remove
                        # remove
                        pad_tensor = torch.LongTensor([src_dict.pad()] * (n_sampled + start_idx)).to(src_tokens)
                        idx_reserved = random.sample(range(start_idx,end_idx),end_idx - n_sampled-start_idx)+[end_idx]
                        idx_reserved.sort()
                        idx_reserved = torch.LongTensor(idx_reserved).to(src_tokens)
                        src_tokens[i] = torch.cat((pad_tensor,s[idx_reserved]),0).to(src_tokens)
                        src_lengths[i] = src_lengths[i] - n_sampled
                    elif rd >= rd2 and rd < 1: # swap
                        # swap
                        if start_idx >= end_idx-1 or n_sampled > end_idx-1-start_idx:
                           continue 
                        origial_src = s.clone()
                        idx_swap1 = torch.LongTensor(random.sample(range(start_idx,end_idx-1),n_sampled)).to(src_tokens)
                        idx_swap2 = idx_swap1 + 1
                        s[idx_swap1] = origial_src[idx_swap2]
                        s[idx_swap2] = origial_src[idx_swap1]
                        src_tokens[i] = s

                elif noise_type == 'insert':
                    idx_list1 = random.sample(range(start_idx,end_idx),n_sampled)
                    idx_list1.sort()
                    idx_list2 = random.sample(range(4,len(src_dict.symbols)),n_sampled)
                    s_list = s.tolist()

                    for insert_id,insert_idx in enumerate(idx_list1):
                        s_list.insert(insert_idx,idx_list2[insert_id])

                    src_tokens_extend[i] = torch.LongTensor(s_list).to(src_tokens)
                    src_lengths[i] = src_lengths[i] + n_sampled
        if (noise_rate != 0.0) and  (noise_type == 'insert'):
            src_tokens = src_tokens_extend      
    return src_tokens,src_lengths,cur_max_noise_rate

class TransformerEncoderRobustAll(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )
