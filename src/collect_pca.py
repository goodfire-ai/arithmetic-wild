"""
For a particular task/model, load the dataset for that task/model. Collect activations
for all layers at all the important token positions, and compute PCA for each combination.

Then visualize the first three PCs for each of the specified `label_variables` (e.g. color by day, number, etc.).
Also save the activations and the SVD results alongside the plots.

Output dir: outputs/pca/{model_name}/{args.task_folder}/features/{key[0]}__{key[1]}.safetensors"

Output structure:
    outputs/pca/{model_name}/{args.task_folder}/
    ├── metadata.json
    ├── features/
    ├── svd/
    └── plots/
        └── {layer}__{position}/
            ├── by_{label_variable}/
            │   └── pc0_vs_pc1_vs_pc2.png
            └── by_{label_variable}/
                └── pc0_vs_pc1_vs_pc2.png
                ...

Usage:
    python scripts/collect_pca.py \
        --task-folder weekdays \
        --model "meta-llama/Llama-3.1-8B" \
"""
import re
import argparse
import os
import shutil
import time

import torch
from collections import defaultdict

from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.jobs.collect_PCA import collect_and_compute_PCA
from causalab.experiments.visualizations.pca_scatter import plot_pca_scatter
from causalab.neural.pipeline import LMPipeline
from utils import load_dataset
from tasks import TASKS
from cl_patch import CounterfactualDataset as CFDataset

def main():
    parser = argparse.ArgumentParser(
        description="PCA analysis for a given task's activations"
    )
    parser.add_argument(
        "--task-folder",
        type=str,
        required=True,
        help="Model name or path (e.g. months)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name or path (default: meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--label-variables",
        type=str,
        nargs="+",
        default=["all"],
        help="Variables to color dots on visualizations by: use 'all' to analyze all variables, or specify one or more positions.",
    )
    parser.add_argument(
        "--token-positions",
        type=str,
        nargs="+",
        default=["all"],
        help="Token position(s) to collect activations at: use 'all' to analyze all positions, or specify one or more positions.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=100,
        help="Number of PCA components to compute (default: 100)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
        help="Path to JSON dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pca/",
        help="Output directory for PCA results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature collection (default: 32)",
    )
    parser.add_argument(
        "--sublayer",
        type=str,
        default="block_output",
        choices=["block_output", "post_attn_resid"],
        help="Where to hook residual stream: block_output (post-layer) or post_attn_resid (after attention, before MLP)",
    )

    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    dataset_path = os.path.join(
        args.dataset_path,
        model_name,
        args.task_folder,
        "filtered_dataset.json"
    )
    # For non-default sublayer, put outputs in a sublayer-specific subdirectory
    if args.sublayer == "block_output":
        output_dir = os.path.join(
            args.output_dir,
            model_name,
            args.task_folder
        )
    else:
        output_dir = os.path.join(
            args.output_dir,
            model_name,
            args.task_folder,
            args.sublayer
        )

    # Load dataset from JSON
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Deduplicate by unique input strings to avoid biasing PCA
    # (many rows share the same base input with different counterfactuals)
    seen_inputs = set()
    unique_indices = []
    for i in range(len(dataset)):
        raw = dataset[i]["input"]["raw_input"]
        if raw not in seen_inputs:
            seen_inputs.add(raw)
            unique_indices.append(i)
    n_before = len(dataset)
    n_unique = len(unique_indices)
    print(f"Deduplicated to {n_unique} unique prompts (removed {n_before - n_unique} duplicates)")

    unique_data = {
        "input": [dataset[i]["input"] for i in unique_indices],
        "counterfactual_inputs": [dataset[i]["counterfactual_inputs"] for i in unique_indices],
    }
    dataset = CFDataset.from_dict(unique_data, id="unique")

    # Load model
    print(f"\nLoading model: {args.model}")
    start_time = time.time()
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=1,
        dtype=torch.bfloat16,
    )
    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.2f}s")

    # get causal model and possible tok positions
    task = re.sub(r'(_no-mtws|_unfiltered)', '', args.task_folder)
    all_token_positions = TASKS[task]["create_token_positions"](pipeline)
    causal_model = TASKS[task]["causal_model"]

    # Select token positions based on argument
    if args.token_positions == ["all"]:
        token_positions = list(all_token_positions.values())
    else:
        token_positions = [all_token_positions[k] for k in args.token_positions]

    print(f"Using token positions: {args.token_positions}")
    print(f"Using sublayer: {args.sublayer}")

    # Build targets for all layers and all token positions
    num_layers = pipeline.model.config.num_hidden_layers

    # For post_attn_resid, layer -1 (embeddings) doesn't make sense - skip it
    if args.sublayer == "post_attn_resid":
        layers = list(range(num_layers))
    else:
        layers = [-1] + list(range(num_layers))

    print(f"\nBuilding targets for {len(layers)} layers x {len(token_positions)} positions...")
    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )
    print(f"Built {len(targets)} targets")

    # actually retrieve the list of label variables we'll use to color points
    if args.label_variables == ["all"]:
        label_variables = [v for v in causal_model.variables if "raw" not in v]
    else:
        for v in args.label_variables:
            assert v in causal_model.variables
        label_variables = args.label_variables

    # Get all of the relevant labels we'll need to color using
    print(f"\nExtracting labels for all variables: {label_variables}")
    all_labels = defaultdict(list)
    for i in range(len(dataset)):
        output = causal_model.run_forward(dataset[i]["input"])
        for var in label_variables:
            all_labels[var].append(output[var])

    # Build list of CounterfactualExample dicts for collect_and_compute_PCA
    # (the new API takes `data` directly instead of a HF dataset path)
    data = [dataset[i] for i in range(len(dataset))]

    print(f"\nRunning PCA analysis...")
    print(f"  n_components: {args.n_components}")
    print(f"  output_dir: {output_dir}")
    print(f"\nThis will generate {len(targets)} targets x {len(label_variables)} label types = {len(targets) * len(label_variables)} plots...")

    start_time = time.time()
    placeholder = list(all_labels.keys())[0]
    result = collect_and_compute_PCA(
        interchange_targets=targets,
        data=data,
        pipeline=pipeline,
        labels=all_labels[placeholder],  # Placeholder; placeholder plots are wiped below
        component_tuples=[(0, 1, 2)],  # Only 3D plots
        n_components=args.n_components,
        output_dir=output_dir,
        batch_size=args.batch_size,
        save_results=True,
        verbose=True,
    )
    collection_elapsed = time.time() - start_time

    # The new API has no `generate_plots=False` switch — it always writes plots
    # when save_results=True. Wipe the placeholder plots so we can regenerate
    # cleanly per-label-variable below.
    placeholder_plots_dir = os.path.join(output_dir, "plots")
    if os.path.isdir(placeholder_plots_dir):
        shutil.rmtree(placeholder_plots_dir)

    # Generate plots for all label variables
    print(f"\n{'=' * 70}")
    print("GENERATING PLOTS FOR ALL LABEL VARIABLES")
    print(f"{'=' * 70}")

    plots_dir = os.path.join(output_dir, "plots")
    component_tuples = [(0, 1, 2)]

    # plot_pca_scatter sorts the legend/color cycle by str(label), which gives
    # alphabetical order ("April, August, ...") or lex-sorted ints ("1, 10, 2, ...").
    # To force canonical ordering, prefix each label with a zero-padded canonical
    # index — `sorted(..., key=str)` then produces canonical order.
    task_cfg = TASKS[task]

    def get_idx_fn(var_name):
        if var_name == "offset":
            return task_cfg["num_to_idx"]
        if var_name in ("input", "output"):
            return task_cfg["inp_to_idx"]
        if var_name == "premod":
            return int
        return None

    def reorder_labels(labels_seq, idx_fn):
        indices = [idx_fn(v) for v in labels_seq]
        pad = max(2, len(str(max(indices))))
        return [f"{idx:0{pad}d}: {v}" for idx, v in zip(indices, labels_seq)]

    for label_var in label_variables:
        labels = all_labels[label_var]
        n_unique = len(set(labels))

        if n_unique > 30:
            print(f"\nSkipping plots for '{label_var}': {n_unique} unique labels (> 30)")
            continue

        idx_fn = get_idx_fn(label_var)
        if idx_fn is not None:
            try:
                labels = reorder_labels(labels, idx_fn)
            except Exception as e:
                print(f"  Warning: could not reorder '{label_var}' ({e}); falling back to default order")

        print(f"\nGenerating plots colored by: {label_var} ({n_unique} unique labels)")

        for key_str, svd_result in result["svd_results_by_target"].items():
            # Create subdirectory for this label variable
            target_plot_dir = os.path.join(plots_dir, key_str, f"by_{label_var}")
            os.makedirs(target_plot_dir, exist_ok=True)

            features = result["features_by_target"][key_str]

            # plot 3d
            plot_pca_scatter(
                features=features,
                svd_result=svd_result,
                labels=labels,
                component_tuples=component_tuples,
                title=f"{key_str} (by {label_var})",
                save_dir=target_plot_dir,
            )

            # plot 2d, just pc0, pc1
            plot_pca_scatter(
                features=features,
                svd_result=svd_result,
                labels=labels,
                component_tuples=[component_tuples[0][:2]],
                title=f"{key_str} (by {label_var})",
                save_dir=target_plot_dir,
            )

    total_elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'=' * 70}")
    print("PCA ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Feature collection time: {collection_elapsed:.2f}s")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"\nVariance explained by first 3 components:")
    print(f"{'=' * 70}")

    # Group by token position for cleaner output
    for pos_name in ["last_token", "number", "day"]:
        print(f"\n{pos_name.upper()}:")
        print("-" * 50)
        for layer in layers:
            key_str = f"{layer}__{pos_name}"
            if key_str in result["svd_results_by_target"]:
                svd_result = result["svd_results_by_target"][key_str]
                var_ratios = svd_result["explained_variance_ratio"]
                total_var = sum(var_ratios[:3])
                print(
                    f"  Layer {layer:2d}: "
                    f"PC0={var_ratios[0]:.1%}, "
                    f"PC1={var_ratios[1]:.1%}, "
                    f"PC2={var_ratios[2]:.1%} "
                    f"(total: {total_var:.1%})"
                )

    print(f"\n{'=' * 70}")
    print(f"Plots saved to: {plots_dir}")
    print(f"  Each target has subdirectories: by_number/, by_day/, by_result_day/")
    print(f"Features saved to: {result['output_paths'].get('features_dir', 'N/A')}")
    print(f"SVD results saved to: {result['output_paths'].get('svd_dir', 'N/A')}")

    return result


if __name__ == "__main__":
    main()