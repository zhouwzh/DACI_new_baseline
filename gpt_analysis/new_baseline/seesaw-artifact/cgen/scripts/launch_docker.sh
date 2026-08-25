#!/bin/bash

docker run -i -t --gpus=all -v .:/workspace/cgen -v "${HOME}"/.cache/huggingface/:/root/.cache/huggingface --shm-size=64g cgen:latest
