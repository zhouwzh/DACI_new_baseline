# Guide

This document is specifically for artifact evaluation of the paper *Seesaw: High-throughput LLM Inference via Model Re-sharding* published in MLSys 2025.

## Prerequisite
- CUDA runtime >= 12.1
- Docker
- NVIDIA container toolkits ([Install instruction](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

## Clone this repo

Please skip this step if you download the artifacts from other sources.

```bash
git clone -b ae https://github.com/soodoshll/cgen.git 
cd cgen
```

The work directory is the one this readme file belong to.

### Build Docker Image

Run this command to build the docker image and download the ShareGPT dataset.

```bash
docker build -t cgen:latest . 
```

The default building options are for L4 GPUs. If you are using a different GPU architecture, you can modify the `TORCH_CUDA_ARCH_LIST` build arg like:

```bash
docker build -t cgen:latest . --build-arg TORCH_CUDA_ARCH_LIST="<your GPU arch>"
```

Please refer to [Wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) and [this comment](https://github.com/ashawkey/stable-dreamfusion/issues/360#issuecomment-2292510049) on how to choose the proper value for `TORCH_CUDA_ARCH_LIST`.

### Launch Experiment

Before the experiment, ensure the access to the Huggingface model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), and save your HuggingFace token as an environment variable `HF_TOKEN`:
```
export HF_TOKEN=<your hf_token>
```

A quick way to run the experiment is through the `benchmark/run_benchmark.sh` script:
```
sh benchmark/run_benchmark.sh
```

This script contains the following command:


```bash
docker run --gpus=all \
    -v .:/workspace/cgen \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    --shm-size 64g \
    cgen:latest \
    python3 benchmark/benchmark_cgen.py \
    benchmark/configs/llama2-7b-2xl4.json \
    --dataset sharegpt \
    --dataset-path /datasets/sharegpt.json \
    --hf-token ${HF_TOKEN}
```

### What need to be modified for more experiments

We provide a minimal example in `benchmark/configs/llama2-7b-2xl4.json`.
To create new experiments, copy this file and modify. 
You can modify arguments including models, parallelization strategies, and the CPU/GPU cache size.
More experiment arguments can be found in `benchmark/run_benchmark.sh`.

**Known limitations**
 - Now we only support Llama family model.
 - Only support transition from larger PP for prefiling to smaller PP for decoding.