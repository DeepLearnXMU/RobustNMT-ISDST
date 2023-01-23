

lang_list="ja ko es ru fr de"

for lang in $lang_list
do 

# en <=> lang
#fairseq-generate data-bin/${lang}-en/  \
#--batch-size 10 --path 1.2B_last_checkpoint.pt  \
#--fixed-dictionary model_dict.128k.txt -s ${lang} -t en \
#--remove-bpe 'sentencepiece' --beam 5 \
#--task translation_multi_simple_epoch \
#--lang-pairs language_pairs_small_models.txt \
#--decoder-langtok --gen-subset test > testsets/generated/big/${lang}-en.gen
echo "lang"
#fairseq-generate data-bin/${lang}-en/  \
#--batch-size 10 --path 1.2B_last_checkpoint.pt \
#--fixed-dictionary model_dict.128k.txt -s en -t ${lang} \
#--remove-bpe 'sentencepiece' --beam 5 \
#--task translation_multi_simple_epoch \
#--lang-pairs language_pairs_small_models.txt \
#--decoder-langtok --gen-subset test > testsets/generated/big/en-${lang}.gen

#bash ../../scripts/compound_split_bleu.sh testsets/generated/big/${lang}-en.gen
#bash ../../scripts/compound_split_bleu.sh testsets/generated/big/en-${lang}.gen

#python /home/miaozhongjian1/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=testsets/generated/big/en-${lang}.gen.sys  --outputs=testsets/generated/big/en-${lang}.gen.sys.spm
#python /home/miaozhongjian1/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=testsets/generated/big/${lang}-en.gen.sys  --outputs=testsets/generated/big/${lang}-en.gen.sys.spm

#sacrebleu testsets/spm.${lang}-en.en -i testsets/generated/${lang}-en.gen.sys.spm -m bleu -b -w 2
#sacrebleu testsets/spm.${lang}-en.${lang} -i testsets/generated/en-${lang}.gen.sys.spm -m bleu -b -w 2

done 

for lang in $lang_list
do

#bash ../../scripts/compound_split_bleu.sh testsets/generated/big/${lang}-en.gen
#bash ../../scripts/compound_split_bleu.sh testsets/generated/big/en-${lang}.gen
#python /home/miaozhongjian1/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=testsets/generated/big/en-${lang}.gen.sys  --outputs=testsets/generated/big/en-${lang}.gen.sys.spm
#python /home/miaozhongjian1/fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=testsets/generated/big/${lang}-en.gen.sys  --outputs=testsets/generated/big/${lang}-en.gen.sys.spm
#echo $lang=>en
#sacrebleu testsets/spm.${lang}-en.en -i testsets/generated/big/${lang}-en.gen.sys.spm -m bleu -b -w 2
#cat testsets/generated/${lang}-en.gen.sys.spm | sacrebleu testsets/spm.${lang}-en.en
#echo en=>$lang
#sacrebleu testsets/spm.${lang}-en.${lang} -i testsets/generated/big/en-${lang}.gen.sys.spm -m bleu -b -w 2
#cat testsets/generated/en-${lang}.gen.sys.spm | sacrebleu testsets/spm.${lang}-en.${lang}

sacrebleu -m chrf --chrf-word-order 2 testsets/spm.${lang}-en.en  < testsets/generated/big/${lang}-en.gen.sys.spm
sacrebleu -m chrf --chrf-word-order 2 testsets/spm.${lang}-en.${lang} < testsets/generated/big//en-${lang}.gen.sys.spm

#sacrebleu -m chrf --chrf-word-order 2 testsets/generated/big/${lang}-en.gen.ref  < testsets/generated/big/${lang}-en.gen.sys
#sacrebleu -m chrf --chrf-word-order 2 testsets/generated/big/en-${lang}.gen.ref  < testsets/generated/big/en-${lang}.gen.sys
echo "==="

done
