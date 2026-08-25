ARG CUDA_VERSION=12.4.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

RUN apt-get update 
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git wget

RUN pip3 install -U pip setuptools wheel
RUN pip3 install vllm==0.5.5

WORKDIR /workspace
COPY . /workspace/cgen

ARG TORCH_CUDA_ARCH_LIST="8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN pip3 install -e cgen/

RUN pip3 install flashinfer==0.1.2 -i https://flashinfer.ai/whl/cu121/torch2.4/

WORKDIR /datasets/
RUN sh /workspace/cgen/benchmark/download_sharegpt.sh /datasets/

WORKDIR /workspace/cgen/
