
seed=64
src=de
tgt=en
# training stage 1, focusing on adversarial examples
fairseq-train data-bin/iwslt14.tokenized.de-en \
--arch robust_transformer_iwslt_de_en_all --share-all-embeddings \
--optimizer adam --lr 0.0005 -s $src -t $tgt \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
--lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion cross_entropy_with_robust_all \
--reg-alpha 1.5 --no-progress-bar --seed ${seed} --fp16 --eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses --eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-update 200000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
--save-dir /path/to/stage1 --kl-direction noise --only-nll noise \
--add-noise --noise-type hybrid  --noise-rate 0.1 --noise-seed ${seed} --is-half-batch \
--apply-prob 1.0 --no-epoch-checkpoints --curriculum-learning \
--curriculum-args '{"max_rate":0.1,"min_rate":0.0,"p":2,"cupdates":10000,"mupdates":200000,"reverse":0}'

# training stage 2, focusing on authentic examples
fairseq-train data-bin/iwslt14.tokenized.de-en \
--arch robust_transformer_iwslt_de_en_all --share-all-embeddings \
--optimizer adam --lr 0.0005 -s $src -t $tgt \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
--lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion cross_entropy_with_robust_all --reg-alpha 1.5 \
--no-progress-bar --seed ${seed} --fp16 --eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses --eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-update 150000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
--save-dir path/to/stage2 --kl-direction clean --only-nll clean \
--add-noise --noise-type hybrid  --noise-rate 0.1 --noise-seed ${seed} --is-half-batch \
--apply-prob 1.0 --no-epoch-checkpoints --curriculum-learning \
--curriculum-args '{"max_rate":0.1,"min_rate":0.0,"p":2,"cupdates":10000,"mupdates":150000,"reverse":0}' \
--finetune-from-model  path/to/stage1/checkpoint_best.pt

# training stage 3, focusing on adversarial examples
fairseq-train data-bin/iwslt14.tokenized.de-en \
--arch robust_transformer_iwslt_de_en_all --share-all-embeddings \
--optimizer adam --lr 0.0005 -s $src -t $tgt \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
--lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion cross_entropy_with_robust_all --reg-alpha 1.5 \
--no-progress-bar --seed ${seed} --fp16 --eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses --eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-update 200000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
--save-dir /path/to/stage1 --kl-direction noise --only-nll noise \
--add-noise --noise-type hybrid  --noise-rate 0.1 --noise-seed ${seed} --is-half-batch \
--apply-prob 1.0 --no-epoch-checkpoints --curriculum-learning \
--curriculum-args '{"max_rate":0.1,"min_rate":0.0,"p":2,"cupdates":10000,"mupdates":200000,"reverse":0}' \
--finetune-from-model  path/to/stage2/checkpoint_best.pt

# training stage 4, focusing on authentic examples
fairseq-train data-bin/iwslt14.tokenized.de-en \
--arch robust_transformer_iwslt_de_en_all --share-all-embeddings \
--optimizer adam --lr 0.0005 -s $src -t $tgt \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
--lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion cross_entropy_with_robust_all \
--reg-alpha 1.5 --no-progress-bar --seed ${seed} --fp16 --eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses --eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-update 150000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
--save-dir path/to/stage2 --kl-direction clean --only-nll clean \
--add-noise --noise-type hybrid  --noise-rate 0.1 --noise-seed ${seed} --is-half-batch \
--apply-prob 1.0 --no-epoch-checkpoints --curriculum-learning \
--curriculum-args '{"max_rate":0.1,"min_rate":0.0,"p":2,"cupdates":10000,"mupdates":150000,"reverse":0}' \
--finetune-from-model  path/to/stage3/checkpoint_best.pt

# training stage 5, focusing on adversarial examples
fairseq-train data-bin/iwslt14.tokenized.de-en \
--arch robust_transformer_iwslt_de_en_all --share-all-embeddings \
--optimizer adam --lr 0.0005 -s $src -t $tgt \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 \
--lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion cross_entropy_with_robust_all --reg-alpha 1.5 \
--no-progress-bar --seed ${seed} --fp16 --eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses --eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--max-update 200000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
--save-dir /path/to/stage1 --kl-direction noise --only-nll noise \
--add-noise --noise-type hybrid  --noise-rate 0.1 --noise-seed ${seed} --is-half-batch \
--apply-prob 1.0 --no-epoch-checkpoints --curriculum-learning \
--curriculum-args '{"max_rate":0.1,"min_rate":0.0,"p":2,"cupdates":10000,"mupdates":200000,"reverse":0}' \
--finetune-from-model  path/to/stage4/checkpoint_best.pt