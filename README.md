# RobustNMT-ISDST
Code for "Towards Robust Neural Machine Translation with Iterative Scheduled Data-Switch Training" (COLING 2022 Oral)
## Requirements

- Python 3.7.0
- CUDA 11.6
- Pytorch 1.12.0
- Fairseq 0.12.2

## Quickstart
Here, we take IWSLT14 De-En translation task as an example.
### Step1: Preprocess

```
# Download and clean the raw data
bash examples/translation/prepare-iwslt14.sh

# Preprocess/binarize the data
mkdir -p data-bin
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref /path/to/iwslt14_deen_data/train \
    --validpref /path/to/iwslt14_deen_data/valid \
    --testpref /path/to/iwslt14_deen_data/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
### Step2: Training
Our training processes are conducted in an iterative manner. The iterative number K is set as 5 for IWSLT14 De-En translation task.
```
bash train_iwslt14-de-en.sh
```
### Step3: Evaluation
```
# generate translations
fairseq-generate data-bin/iwslt14.tokenized.de-en \
--path /path/to/stage5/checkpoint_best.pt \
--beam 5 --remove-bpe > res.out
# calculate BLEU score
bash scripts/compound_split_bleu.sh res.out
```
## Citation
```
@inproceedings{miao2022towards,
  title={Towards Robust Neural Machine Translation with Iterative Scheduled Data-Switch Training},
  author={Miao, Zhongjian and Li, Xiang and Kang, Liyan and Zhang, Wen and Zhou, Chulun and Chen, Yidong and Wang, Bin and Zhang, Min and Su, Jinsong},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={5266--5277},
  year={2022}
}
```
