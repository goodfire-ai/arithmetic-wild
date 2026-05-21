"""Cache MLP neuron activations (gate, up, both) for all prompts across tasks.

Saves to: {output_dir}/{model_name}/{task}_L{layer}/
  - gate_mlp_acts.pt  (len(nums), len(inps), intermediate_size)
  - up_mlp_acts.pt
  - both_mlp_acts.pt
  - metadata.json      (inps list, nums list, task name, layer, model)

Usage:
    python src/cache_neuron_activations.py --tasks weekdays months hours addition --layer 18
"""

import argparse
import json
import os

import torch
from nnsight import LanguageModel

# Task configurations: (inps, nums, template)
TASK_CONFIGS = {}


def _load_task_config(task_name: str):
    """Lazily load task config to avoid importing all tasks upfront."""
    if task_name in TASK_CONFIGS:
        return TASK_CONFIGS[task_name]

    if task_name == "weekdays":
        from tasks.weekdays.causal_models import DAYS, OFFSETS, TEMPLATE
        cfg = (DAYS, OFFSETS, TEMPLATE)
    elif task_name == "months":
        from tasks.months.causal_models import MONTHS, OFFSETS, TEMPLATE
        cfg = (MONTHS, OFFSETS, TEMPLATE)
    elif task_name == "hours":
        from tasks.hours.causal_models import HOURS, OFFSETS, TEMPLATE
        cfg = (HOURS, OFFSETS, TEMPLATE)
    elif task_name == "addition":
        from tasks.addition.causal_models import NUMBERS, TEMPLATE
        cfg = (NUMBERS, NUMBERS, TEMPLATE)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    TASK_CONFIGS[task_name] = cfg
    return cfg


def build_prompts(inps, nums, template):
    """Build dict mapping (num, inp) -> prompt string."""
    prompts = {}
    for num in nums:
        for inp in inps:
            prompts[(inp, num)] = template.format(offset=num, input=inp)
    return prompts


def collect_mlp_activations(
    model: LanguageModel,
    mlp_layer: int,
    prompts: dict,
    inps: list,
    nums: list,
) -> tuple:
    """Collect MLP neuron activations for all prompts.

    Uses 0-based enumerate indexing for both axes.

    Returns:
        (both_mlp_acts, gate_mlp_acts, up_mlp_acts) each of shape
        (len(nums), len(inps), intermediate_size)
    """
    inp_to_i = {inp: i for i, inp in enumerate(inps)}
    num_to_i = {num: i for i, num in enumerate(nums)}

    intermediate_size = model.config.intermediate_size

    gate_mlp_acts = torch.zeros((len(nums), len(inps), intermediate_size), device="cuda")
    up_mlp_acts = torch.zeros((len(nums), len(inps), intermediate_size), device="cuda")
    both_mlp_acts = torch.zeros((len(nums), len(inps), intermediate_size), device="cuda")

    with torch.no_grad():
        for (inp, num), prompt in prompts.items():
            with model.trace(prompt):
                gate_neurons = model.model.layers[mlp_layer].mlp.source.self_act_fn_0.output[0, -1].save()
                up_neurons = model.model.layers[mlp_layer].mlp.up_proj.output[0, -1].save()
                both_neurons = model.model.layers[mlp_layer].mlp.down_proj.input[0, -1].save()

            ni, ii = num_to_i[num], inp_to_i[inp]
            gate_mlp_acts[ni, ii] = gate_neurons
            up_mlp_acts[ni, ii] = up_neurons
            both_mlp_acts[ni, ii] = both_neurons

    return both_mlp_acts, gate_mlp_acts, up_mlp_acts


def parse_args():
    parser = argparse.ArgumentParser(description="Cache MLP neuron activations for tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["weekdays", "months", "hours", "addition"],
        choices=["weekdays", "months", "hours", "addition"],
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--model-name", type=str, default="Llama-3.1-8B")
    parser.add_argument("--output-dir", type=str, default="outputs/neuron_activations")
    return parser.parse_args()


def main():
    args = parse_args()

    model_id = f"meta-llama/{args.model_name}"
    print(f"Loading model: {model_id}")
    model = LanguageModel(model_id, device_map="cuda", dispatch=True, torch_dtype=torch.bfloat16)
    print(f"Intermediate size: {model.config.intermediate_size}")

    for task_name in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name}, Layer: {args.layer}")
        print(f"{'=' * 60}")

        inps, nums, template = _load_task_config(task_name)
        prompts = build_prompts(inps, nums, template)
        print(f"Prompts: {len(prompts)} ({len(nums)} nums x {len(inps)} inps)")

        both, gate, up = collect_mlp_activations(model, args.layer, prompts, inps, nums)

        # Save
        out_dir = os.path.join(args.output_dir, args.model_name, f"{task_name}_L{args.layer}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(gate.cpu(), os.path.join(out_dir, "gate_mlp_acts.pt"))
        torch.save(up.cpu(), os.path.join(out_dir, "up_mlp_acts.pt"))
        torch.save(both.cpu(), os.path.join(out_dir, "both_mlp_acts.pt"))

        metadata = {
            "inps": list(inps),
            "nums": list(nums),
            "task": task_name,
            "layer": args.layer,
            "model_name": args.model_name,
            "shape": list(both.shape),
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved to {out_dir}/")
        print(f"  gate_mlp_acts.pt: {gate.shape}")
        print(f"  up_mlp_acts.pt:   {up.shape}")
        print(f"  both_mlp_acts.pt: {both.shape}")


if __name__ == "__main__":
    main()
