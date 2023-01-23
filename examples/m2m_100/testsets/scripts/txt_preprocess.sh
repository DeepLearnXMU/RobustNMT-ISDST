src=fr
fairseq-preprocess \
  --source-lang fr \
  --target-lang en \
  --testpref spm.fr-en \
  --thresholdsrc 0 --thresholdtgt 0 \
  --destdir ../data-bin/fr-en \    
  --srcdict ../data_dict.128k.txt \
  --tgtdict ../data_dict.128k.txt
