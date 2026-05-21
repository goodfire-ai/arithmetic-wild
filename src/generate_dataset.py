"""
Generate and filter counterfactual datasets for tasks in the `tasks` folder. 

This script:
1. Generates a random counterfactual dataset of a given size
2. Filters the dataset using a language model
3. Reports filter accuracy
4. Saves the filtered dataset

Note that filter accuracy is not the same as base accuracy for the task, because 
it requires both randomly-sampled examples to be correct to keep it. 

Usage:
    python scripts/generate_and_filter \
        --task months \
        --model meta-llama/Llama-3.1-8B \
"""

import argparse
import json
import os
import time
from typing import Dict

import torch
import numpy as np 
import random 

from causalab.causal.causal_model import CausalModel
from causalab.experiments.filter import filter_dataset
from causalab.neural.pipeline import LMPipeline
from datasets import Dataset 

from tasks import TASKS, random_counterfactual
from utils import metric, CounterfactualDataset

def all_are_single_token(sample: Dict, pipeline: LMPipeline, causal_model: CausalModel, vars_to_check, check_raw_output: bool = True) -> bool:
    """
    If any of the causal variables are multi-token words, return False. 
    If check_raw_output=True, also checks to see if the expected output is more than a single token. 

    Args:
        sample: dict with 'input' and 'counterfactual_inputs' keys
        pipeline: LM pipeline with tokenizer
        causal_model: causal model for this task 
        vars_to_check: list of str variables that we want to be single token 
        check_raw_output: also check if the expected raw output is a single token. 

    Returns:
        True if variables we'd want to intervene on (and possibly raw_output) are all single-token. 
    """
    if check_raw_output:
        vars_to_check.append("raw_output")

    # Check base input
    ran_forward = causal_model.run_forward(sample["input"])
    for label, var in ran_forward.items():
        if label in vars_to_check:
            # Use add_special_tokens=False to exclude BOS/EOS tokens
            # NOTE: assumes that the variable has a preceding space. 
            withspace = var if label == "raw_output" else " " + var 
            toks = pipeline.load(
                {"raw_input": withspace},
                add_special_tokens=False,
                no_padding=True
            )["input_ids"][0]
            if len(toks) > 1:
                return False 


    # Check all counterfactual inputs
    for cf_input in sample["counterfactual_inputs"]:
        ran_forward = causal_model.run_forward(cf_input)
        for label, var in ran_forward.items():
            if label in vars_to_check:
                # Use add_special_tokens=False to exclude BOS/EOS tokens
                toks = pipeline.load(
                    {"raw_input": var},
                    add_special_tokens=False,
                    no_padding=True
                )["input_ids"][0]
                if len(toks) > 1:
                    return False 

    return True

def generate_dataset(causal_model: CausalModel, size: int, filter = None) -> CounterfactualDataset:
    """
    Generate a random counterfactual dataset of given size.

    Args:
        causal_model: CausalModel to generate examples for 
        size: Number of examples to generate
        filter: if not None, provides condition to filter when sampling

    Returns:
        CounterfactualDataset with random counterfactual pairs
    """
    print(f"Generating {size} counterfactual examples...")
    start_time = time.time()

    dataset = CounterfactualDataset.from_sampler(
        size=size,
        counterfactual_sampler=lambda: random_counterfactual(causal_model),
        filter=filter 
    )

    elapsed = time.time() - start_time
    print(f"Generated {len(dataset)} examples in {elapsed:.2f}s")

    # return dataset 

    # run causal model on each example so it has raw_output
    ran_dataset = [{} for _ in range(len(dataset))]
    for i in range(len(dataset)):
        ran_dataset[i]['input'] = causal_model.run_forward(dataset[i]['input'])
        temp = []
        for x in dataset[i]['counterfactual_inputs']:
            temp.append(causal_model.run_forward(x))
        ran_dataset[i]['counterfactual_inputs'] = temp 

    return ran_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate and filter for a given task"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g. `weekdays`)",
        choices=TASKS.keys()
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name or path (default: meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1265,
        help="Random seed for dataset generation (default: 177)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for filtering (default: 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/",
        help="Output directory for filtered dataset (default: datasets)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
        help="Maximum new tokens to generate (default: 5)",
    )
    parser.add_argument(
        "--no-mtws",
        action="store_true",
        default=False,
        help="Whether to filter out variables that contain multiple tokens (default: False)",
    )
    parser.add_argument(
        "--validate-counterfactuals",
        action="store_true",
        default=True,
        help="Also validate counterfactual inputs (default: True)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=4096,
        help="Desired number of examples after filtering. Keeps generating+filtering batches until this many pass. "
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Set up task 
    causal_model = TASKS[args.task]["causal_model"]

    # Load model
    print(f"\nLoading model: {args.model}")
    print(f"  dtype: {args.dtype}")
    print(f"  max_new_tokens: {args.max_new_tokens}")

    start_time = time.time()
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
    )
    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.2f}s")

    # set up MTW filter
    filter = None 
    if args.no_mtws:
        filter = lambda sample: all_are_single_token(
            sample, 
            pipeline, 
            causal_model,  
            TASKS[args.task]["single_token_important"],
            check_raw_output=True
        )


    # Iterative mode: keep generating + filtering until we have enough
    target = args.target_size
    batch_gen_size = 1000
    all_filtered_inputs = []
    all_filtered_counterfactuals = []
    total_generated = 0
    start_time = time.time()

    print(f"\nTarget size: {target}, generating in batches of {batch_gen_size}...")
    print(f"  validate_counterfactuals: {args.validate_counterfactuals}")
    print(f"  no_mtws: {args.no_mtws}")

    batch_num = 0
    while len(all_filtered_inputs) < target:
        batch_num += 1
        remaining = target - len(all_filtered_inputs)
        print(f"\n--- Batch {batch_num} (have {len(all_filtered_inputs)}/{target}, need {remaining} more) ---")

        dataset = generate_dataset(causal_model, batch_gen_size, filter)
        total_generated += len(dataset)

        # filtered_batch, batch_base_acc = filter_dataset(
        filtered_batch = filter_dataset(
            dataset=dataset,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=metric,
            batch_size=args.batch_size,
            validate_counterfactuals=args.validate_counterfactuals,
            # return_base_accuracy=True
        )

        for ex in filtered_batch:
            if len(all_filtered_inputs) >= target:
                break
            all_filtered_inputs.append(ex["input"])
            all_filtered_counterfactuals.append(ex["counterfactual_inputs"])

        print(f"  Batch yielded {len(filtered_batch)} filtered examples")

    elapsed = time.time() - start_time
    filtered_dataset = CounterfactualDataset.from_dict(
        {"input": all_filtered_inputs, "counterfactual_inputs": all_filtered_counterfactuals},
        id="filtered_target",
    )
    original_size = total_generated
    filtered_size = len(filtered_dataset)
    filter_accuracy = filtered_size / original_size if original_size > 0 else 0.0

    print(f"\n{'=' * 60}")
    print("FILTER RESULTS (target-size mode)")
    print(f"{'=' * 60}")
    print(f"Target size:            {target}")
    print(f"Total generated:        {original_size}")
    print(f"Batches needed:         {batch_num}")
    print(f"Final dataset size:     {filtered_size}")
    print(f"Overall filter rate:    {filter_accuracy:.2%} ({filtered_size}/{original_size})")
    print(f"Total time:             {elapsed:.2f}s")
    print(f"{'=' * 60}")

    # Save filtered dataset
    model_name = args.model.split('/')[-1]
    dset_name = args.task
    dset_name += "_no-mtws" if args.no_mtws else ""
    output_dir = os.path.join(args.output_dir, model_name, dset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    filtered_data = {
        "input": [example["input"] for example in filtered_dataset],
        "counterfactual_inputs": [
            example["counterfactual_inputs"] for example in filtered_dataset
        ],
    }
    dataset_path = os.path.join(output_dir, "filtered_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"\nFiltered dataset saved to: {dataset_path}")

    # Save metadata
    metadata = {
        "model": args.model,
        "original_size": original_size,
        "filtered_size": filtered_size,
        "filter_accuracy": filter_accuracy,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "validate_counterfactuals": args.validate_counterfactuals,
        "dtype": args.dtype,
        "filtering_time_seconds": elapsed,
        "no_mtws" : args.no_mtws,
    }
    metadata_path = os.path.join(output_dir, "filter_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
