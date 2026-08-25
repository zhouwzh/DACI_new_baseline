#!/bin/bash

docker run --gpus=all \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --shm-size 64g \
    cgen:latest \
    python3 benchmark/benchmark_cgen.py \
    benchmark/configs/llama2-7b-2xl4.json \
    --dataset sharegpt \
    --dataset-path /datasets/sharegpt.json \
    --hf-token ${HF_TOKEN}
