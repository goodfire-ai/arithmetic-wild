"""
Script that allows you to patch within the union of DAS output subspaces from one task to another. 
You can also patch within the target or source subspace by specifying --subspace. 

We've got some Claude code business here to save things in an apparently-efficient way. 

If you just want to do cross-task patching for a couple examples, check out `cross_task_patch.ipynb`. 
"""
import os
import glob
import json
import torch
import argparse
import itertools
import matplotlib.pyplot as plt

from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
from nnsight import LanguageModel
from collections import defaultdict
from tasks import TASKS
from utils import load_subspace_hf, THREE_CYCLES

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=18,
        help="which result_out subspace to use"
    )
    parser.add_argument(
        "--source-task",
        type=str,
        default="months",
        choices=TASKS,
        help="patch from all of these prompts"
    )
    parser.add_argument(
        "--subspace",
        type=str,
        default="union-source-target",
        choices=["target-subspace", "source-subspace", "union-source-target"],
        help="patch from all of these prompts"
    )
    parser.add_argument(
        "--target-task",
        type=str,
        default="addition",
        choices=TASKS,
        help="patch into all of these prompts"
    )
    parser.add_argument(
        "--hide-source-labels",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=None,
        help="batch size for target prompts (default: all at once)"
    )
    parser.add_argument(
        "--no-limit-source-addition",
        action="store_true",
        default=False,
        help="disable default filtering of addition prompts with premod > THREE_CYCLES of the other task"
    )
    parser.add_argument(
        "--no-limit-target-addition",
        action="store_true",
        default=False,
        help="disable default filtering of target addition prompts to sum < 100"
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=None,
        help="which chunk of source prompts to process (0-indexed)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="total number of chunks to split source prompts into"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="flush accumulated tensors to a new shard file every N source prompts"
    )
    return parser.parse_args()


def shard_name(chunk_idx, shard_idx):
    if chunk_idx is not None:
        return f"chunk_{chunk_idx}_shard_{shard_idx}.safetensors"
    return f"shard_{shard_idx}.safetensors"


def load_cfg_subspace(task, layer):
    return load_subspace_hf(task, "output", layer)

def main():
    args = parse_args()

    subdir = args.subspace
    suffixes = []
    if args.no_limit_source_addition:
        suffixes.append("no_limit_src_add")
    if args.no_limit_target_addition:
        suffixes.append("no_limit_tgt_add")
    suffix = "_" + "_".join(suffixes) if suffixes else ""
    save_dir = f"outputs/ood_patching/{subdir}/{args.source_task}->{args.target_task}/L{args.layer}{suffix}"
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    # determine metadata filename
    if args.chunk_idx is not None:
        meta_filename = f"chunk_{args.chunk_idx}_metadata.json"
    else:
        meta_filename = "metadata.json"

    # load model
    model = LanguageModel("meta-llama/Llama-3.1-8B", device_map='cuda')

    with torch.no_grad():
        # if patching months->hours, patch within the *hours* subspace. 
        if args.subspace == "target-subspace":
            subspace = load_cfg_subspace(args.target_task, args.layer)
        elif args.subspace == "source-subspace":
            subspace = load_cfg_subspace(args.source_task, args.layer)
        elif args.subspace == "union-source-target":
            src = load_cfg_subspace(args.source_task, args.layer)
            tgt = load_cfg_subspace(args.target_task, args.layer)
            subspace, _, _ = torch.linalg.svd(torch.cat([src, tgt], dim=1), full_matrices=False)

        this_proj = (subspace @ subspace.T).bfloat16()
        else_proj = torch.eye(this_proj.shape[0], device=this_proj.device, dtype=this_proj.dtype) - this_proj 

    # generate all possible source and target prompts 
    task_prompts = defaultdict(list)
    for task in [args.source_task, args.target_task]:
        causal_model = TASKS[task]["causal_model"]
        inp_to_idx = TASKS[task]["inp_to_idx"]
        num_to_idx = TASKS[task]["num_to_idx"]
        
        input_vars = causal_model.inputs
        input_value_lists = [causal_model.values[v] for v in input_vars]
        result_var = causal_model.parents["raw_output"][0]

        for combo in itertools.product(*input_value_lists):
            inp = dict(zip(input_vars, combo))
            result = causal_model.run_forward(inp)
            prompt_info = {**result, **inp}
            if "premod" not in prompt_info.keys():
                prompt_info["premod"] = inp_to_idx(prompt_info["input"]) + num_to_idx(prompt_info["offset"])
            task_prompts[task].append(prompt_info)
    
    # filter addition prompts to THREE_CYCLES of the other task
    if not args.no_limit_source_addition:
        if args.source_task == "addition" and args.target_task in THREE_CYCLES.keys():
            limit = THREE_CYCLES[args.target_task]
            before = len(task_prompts["addition"])
            task_prompts["addition"] = [p for p in task_prompts["addition"] if p["premod"] <= limit]
            print(f"Filtered addition source prompts: {before} -> {len(task_prompts['addition'])} (premod <= {limit})")
    
    # by default limit target addition to always limit based on THREE_CYCLES of the source task 
    if not args.no_limit_target_addition:
        if args.target_task == "addition":
            limit = THREE_CYCLES[args.source_task]
            before = len(task_prompts["addition"])
            task_prompts["addition"] = [p for p in task_prompts["addition"] if p["premod"] < limit]
            print(f"Filtered addition target prompts: {before} -> {len(task_prompts['addition'])} (premod <= {limit})")

    # all possible answer tokens across source and target tasks
    possible_output_tokens = []
    if args.hide_source_labels:
        which_tasks = [args.target_task]
    else:
        which_tasks = [args.source_task, args.target_task]

    for task in set(which_tasks):
        causal_model = TASKS[task]["causal_model"]
        result_var = causal_model.parents["raw_output"][0]
        labels = causal_model.values[result_var]
        base = task.split("-")[0]
        prefix = "" if base == "hours" else " "
        for l in labels:
            tid = model.tokenizer(prefix + l, add_special_tokens=False)["input_ids"][0]
            possible_output_tokens.append(tid)
    
    # set up target prompts, optionally batched
    target_prompts = [p["raw_input"] for p in task_prompts[args.target_task]]
    tbs = args.target_batch_size or len(target_prompts)
    target_batches = [target_prompts[i:i+tbs] for i in range(0, len(target_prompts), tbs)]

    # compute clean probs once (same for every source prompt)
    clean_probs_list = []
    with torch.no_grad():
        for batch in target_batches:
            with model.trace(batch):
                cp = model.output.logits[:, -1].softmax(dim=-1).save()
            clean_probs_list.append(cp.detach().cpu())
    clean_probs = torch.cat(clean_probs_list, dim=0)

    # save clean_probs once in its own file
    clean_probs_path = os.path.join(save_dir, "clean_probs.safetensors")
    if not os.path.exists(clean_probs_path):
        save_file({"clean_probs": clean_probs}, clean_probs_path)

    # scan existing shards for completed keys (crash resume)
    completed_keys = set()
    shard_pattern = os.path.join(save_dir, shard_name(args.chunk_idx, "*"))
    existing_shards = sorted(glob.glob(shard_pattern))
    next_shard_idx = len(existing_shards)
    for sp in existing_shards:
        with safe_open(sp, framework="pt") as f:
            completed_keys.update(f.keys())
    if completed_keys:
        print(f"Resuming: {len(completed_keys)} keys already saved in {len(existing_shards)} shards")

    # track all keys (including already-completed ones) for metadata
    all_keys = list(completed_keys)

    # accumulate patched probs for current shard
    all_results = {}

    # chunk source prompts for parallel SLURM jobs
    source_prompts = task_prompts[args.source_task]
    if args.chunk_idx is not None and args.num_chunks is not None:
        chunk_size = len(source_prompts) // args.num_chunks
        start = args.chunk_idx * chunk_size
        end = start + chunk_size if args.chunk_idx < args.num_chunks - 1 else len(source_prompts)
        source_prompts = source_prompts[start:end]
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: processing source prompts {start}-{end}")

    # for each source prompt, patch into all the target prompts and measure prob. deltas
    for prompt_info in tqdm(source_prompts):
        src_prompt = prompt_info["raw_input"]

        desc = f"{prompt_info['offset']}_{prompt_info['input']}_premod{prompt_info['premod']}"
        answer = prompt_info["output"]
        key = f"{answer}/{desc}"

        # skip if already saved in a previous shard
        if key in completed_keys:
            continue

        subdir_answer = os.path.join(save_dir, answer)
        os.makedirs(subdir_answer, exist_ok=True)

        with torch.no_grad():
            # get source state
            with model.trace(src_prompt):
                src = model.model.layers[args.layer].output[:, -1].save()

            # get patched target prompt logits (in batches)
            patch_probs_list = []
            for batch in target_batches:
                with model.trace(batch):
                    bse = model.model.layers[args.layer].output[:, -1]
                    patched = (src @ this_proj) + (bse @ else_proj)
                    model.model.layers[args.layer].output[:, -1] = patched
                    pp = model.output.logits[:, -1].softmax(dim=-1).save()
                patch_probs_list.append(pp.detach().cpu())

            # accumulate raw patched probs for safetensors
            patch_probs = torch.cat(patch_probs_list, dim=0)  # (n_target, vocab_size)
            all_results[key] = patch_probs
            all_keys.append(key)

            # we used to print and plot mean diffs for .txt and .png, but now we do patched probs.
            # mean_diff = (patch_probs - clean_probs).mean(dim=0)
            to_print = patch_probs.mean(dim=0).float()

            s = ""
            for p, t in zip(*torch.topk(to_print, k=30)):
                s += f"{model.tokenizer.decode(t)}\tp={p:.3f}\n"
            with open(os.path.join(subdir_answer, f"{desc}_patched_probs.txt"), "w") as f:
                f.write(s)

            target_tok_diffs = to_print[possible_output_tokens]
            fig, ax = plt.subplots(figsize=(15, 3))
            ax.bar([model.tokenizer.decode(t) for t in possible_output_tokens], target_tok_diffs)
            ax.tick_params(axis='x', rotation=90)
            ax.set_title(f"L{args.layer} Patching from {args.source_task}->{args.target_task}, Patched Probs avg. across {args.target_task} prompts\n{repr(src_prompt)} {answer}")
            ax.set_xlabel("Output Token")
            ax.set_ylabel("P_patched(token)")
            ax.axhline(0, color='black', linestyle='dashed')
            plt.tight_layout()
            plt.savefig(os.path.join(subdir_answer, f"{desc}.png"))
            plt.close(fig)

        # periodic flush to shard file
        if len(all_results) >= args.save_every:
            shard_path = os.path.join(save_dir, shard_name(args.chunk_idx, next_shard_idx))
            save_file(all_results, shard_path)
            print(f"Saved shard {next_shard_idx} with {len(all_results)} keys to {shard_path}")
            next_shard_idx += 1
            all_results = {}

    # flush remaining results
    if all_results:
        shard_path = os.path.join(save_dir, shard_name(args.chunk_idx, next_shard_idx))
        save_file(all_results, shard_path)
        print(f"Saved shard {next_shard_idx} with {len(all_results)} keys to {shard_path}")

    # collect all shard filenames
    all_shard_files = sorted(glob.glob(os.path.join(save_dir, shard_name(args.chunk_idx, "*"))))
    all_shard_filenames = [os.path.basename(f) for f in all_shard_files]

    # write metadata (all keys across all shards)
    metadata = {
        "source_task": args.source_task,
        "target_task": args.target_task,
        "layer": args.layer,
        "subspace": args.subspace,
        "n_target_prompts": len(target_prompts),
        "possible_output_tokens": [int(t) for t in possible_output_tokens],
        "target_prompts": target_prompts,
        "keys": all_keys,
        "shard_files": all_shard_filenames,
    }
    if args.chunk_idx is not None:
        metadata["chunk_idx"] = args.chunk_idx
        metadata["num_chunks"] = args.num_chunks
    with open(os.path.join(save_dir, meta_filename), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()