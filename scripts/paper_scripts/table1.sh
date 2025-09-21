#!/bin/bash
version=table1
ngram=13

for model in "pythia-160m" "pythia-1.4b" "pythia-2.8b" "pythia-6.9b"
do
    for subset in "wikipedia_(en)" "github" "pile_cc" "pubmed_central" "arxiv" "dm_mathematics" "hackernews"
    do
        python run.py \
            --config configs/mi.json \
            --base_model "EleutherAI/${model}" \
            --specific_source ${subset}_ngram_${ngram}_\<0.8_truncated \
            --output_name $version
    done
done

