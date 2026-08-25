#!/bin/bash

download_dir=${1:-.}
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json\
 -O "${download_dir}"/sharegpt.json