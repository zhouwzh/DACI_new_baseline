from typing import List, Optional, Tuple

import argparse
import transformers
import torch
import ray
import os

from cgen.engine import LLMEngine
from cgen.utils import logger
from cgen.config import Plan 

import sys
import time
import random

import json
from dataload import load_dataset, constant_length


def warmup(engine):
    requests = constant_length(512, engine.tokenizer, output_len=1)
    inputs = [x[0] for x in requests]
    max_tokens = [x[1] + x[2] for x in requests]
    engine.run(
        inputs=inputs,
        max_tokens=max_tokens
    )


def run_cgen(
    requests: List[Tuple[str, int, int]],
    plan: Plan,
    prin_output: bool = False,
    log_level: str = "INFO"
) -> float:
    engine = LLMEngine(
        plan.model, plan.tokenizer,plan,
        log_level=log_level,
    )
    # # print("warm up...")
    # warmup(engine)
    # print("warm up finished")
    engine._clear_timing()
    inputs = [x[0] for x in requests]
    max_tokens = [x[1] + x[2] for x in requests]
    out, dur = engine.run(
        inputs=inputs,
        max_tokens=max_tokens
        )
    engine.join()
    if prin_output:
        for i, o in enumerate(out):
            print(i, o)
    return dur
    

def launch(args):
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    with open(args.config) as f:
        plan = Plan.model_validate(json.load(f))
    tokenizer = transformers.AutoTokenizer.from_pretrained(plan.tokenizer)
    datasets = load_dataset(
        args.dataset,
        args.num_requests,
        tokenizer,
        dataset_path = args.dataset_path,
        input_len = args.input_len,
        output_len = args.output_len
    ) 

    if args.sort_prompts:
        datasets.sort(key=lambda x: x[1])
    dur = run_cgen(datasets, plan, args.print_output, log_level=args.log_level)
    print(f"Total time: {dur:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument("--dataset", type=str, default="sharegpt", help="Dataset used for benchmarking. can be sharegpt, cnn, arxiv, constant.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the sharegpt dataset if it is used.")
    parser.add_argument("--num-requests", type=int, default=500, help="Number of requests sent to the engine.")

    parser.add_argument("--log-level", type=str, default='INFO', help="Logger level")
    parser.add_argument("--sort-prompts", action="store_true", help="Sort the prompt by length. Benificial for pipeline parallelism.")
    parser.add_argument("--dummy-weights", action="store_true", help="Use dummy weights for debugging purpose.")
    parser.add_argument("--print-output", action='store_true', help="Print generation results after benchmarking.")
    parser.add_argument("--input-len", type=int, default=1000, help="Number of prompt tokens in a request(when dataset=constant)")
    parser.add_argument("--output-len", type=int, default=100, help="Number of generated tokens (when dataset=constant)")

    parser.add_argument("--hf-token", type=str, required=False, help="HuggingFace token")
    args = parser.parse_args()
    launch(args)