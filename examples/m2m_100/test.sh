

lang_list="ja ko es ru fr de"

for lang in $lang_list
do 

# en <=> lang
#CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/${lang}-en/  \
#--batch-size 120 --path 418M_last_checkpoint.pt \
#--fixed-dictionary model_dict.128k.txt -s ${lang} -t en \
#--remove-bpe 'sentencepiece' --beam 5 \
#--task translation_multi_simple_epoch \
#--lang-pairs language_pairs_small_models.txt \
#--decoder-langtok --gen-subset test > testsets/generated/${lang}-en.gen

#CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/${lang}-en/  \
#--batch-size 120 --path 418M_last_checkpoint.pt \
#--fixed-dictionary model_dict.128k.txt -s en -t ${lang} \
#--remove-bpe 'sentencepiece' --beam 5 \
#--task translation_multi_simple_epoch \
#--lang-pairs language_pairs_small_models.txt \
#--decoder-langtok --gen-subset test > testsets/generated/en-${lang}.gen

#bash ../../scripts/compound_split_bleu.sh testsets/generated/${lang}-en.gen
#bash ../../scripts/compound_split_bleu.sh testsets/generated/en-${lang}.gen

#python /home/miaozhongjian1/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=testsets/generated/en-${lang}.gen.sys  --outputs=testsets/generated/en-${lang}.gen.sys.spm
#python /home/miaozhongjian1/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=testsets/generated/${lang}-en.gen.sys  --outputs=testsets/generated/${lang}-en.gen.sys.spm

#sacrebleu testsets/spm.${lang}-en.en -i testsets/generated/${lang}-en.gen.sys.spm -m bleu -b -w 2
#sacrebleu testsets/spm.${lang}-en.${lang} -i testsets/generated/en-${lang}.gen.sys.spm -m bleu -b -w 2
echo ""

done 

for lang in $lang_list
do

#bash ../../scripts/compound_split_bleu.sh testsets/generated/${lang}-en.gen
#bash ../../scripts/compound_split_bleu.sh testsets/generated/en-${lang}.gen

#echo $lang=>en
# sacrebleu testsets/spm.${lang}-en.en -i testsets/generated/${lang}-en.gen.sys.spm -m bleu -b -w 2
#cat testsets/generated/${lang}-en.gen.sys.spm | sacrebleu testsets/spm.${lang}-en.en
#echo en=>$lang
#sacrebleu testsets/spm.${lang}-en.${lang} -i testsets/generated/en-${lang}.gen.sys.spm -m bleu -b -w 2
#cat testsets/generated/en-${lang}.gen.sys.spm | sacrebleu testsets/spm.${lang}-en.${lang}

#sacrebleu -m chrf --chrf-word-order 2 testsets/spm.${lang}-en.en  < testsets/generated/${lang}-en.gen.sys.spm
#sacrebleu -m chrf --chrf-word-order 2 testsets/spm.${lang}-en.${lang} < testsets/generated/en-${lang}.gen.sys.spm

sacrebleu -m chrf --chrf-word-order 2 testsets/generated/${lang}-en.gen.ref  < testsets/generated/${lang}-en.gen.sys
sacrebleu -m chrf --chrf-word-order 2 testsets/generated/en-${lang}.gen.ref  < testsets/generated/en-${lang}.gen.sys


done
