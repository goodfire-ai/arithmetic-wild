#!/usr/bin/env python3
"""Train DAS on a particular task. By default trains for all token positions and layers.
Default is random initialization, but in the paper we use `random-pca` (initialize within PCs that explain \geq 90 of variance.)

Usage: python scripts/train_das.py \
    --task-folder months \
    --model meta-llama/Llama-3.1-8B \
    --target-variable input # 'input', 'output', or 'offset'

Note: this script only trains on layer outputs: we also train on sublayer outputs, for which
you must do some hacking to hook into the residual stream immediately after the attention sublayer 
output has been added back into the residual stream. 
"""
import re
import os
import json
import hashlib
import argparse
import tempfile
import random
import time
import numpy as np
from datetime import datetime

import torch
from safetensors.torch import load_file

from causalab.neural.pipeline import LMPipeline
from causalab.neural import SubspaceFeaturizer
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.jobs.DAS_grid import train_DAS
from causalab.neural.pyvene_core.collect import compute_svd

from tasks import TASKS
from utils import metric, load_dataset
from cl_patch import CounterfactualDataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DAS on a task in the `tasks` folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task-folder",
        type=str,
        required=True,
        help="Should correspond to a dataset folder datasets/[model]/*task-folder*"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--target-variable",
        type=str,
        required=True,
        help="Target causal variable to localize",
    )
    parser.add_argument(
        "--token-positions",
        type=str,
        nargs="+",
        default=["all"],
        help="Token position(s) to analyze: use 'all' to analyze all positions, or specify one or more positions.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="random",
        choices=["random", "pca", "random-pca"],
        help="whether to initialize subspace randomly, with principal components (computed with `pca_analysis.py`), or randomly within the top PCs capturing 90%% variance",
    )
    parser.add_argument(
        "--subspace-dim",
        type=int,
        default=8,
        help="Dimension of subspace learned for SubspaceFeaturizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8,
        help="Random seed for this training run.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
        help="Path to filtered dataset JSON (from generate_and_filter.py)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/das/",
        help="Path to filtered dataset JSON (from generate_and_filter.py)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (rest for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16, # could also do bsz=32, epochs=16
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer(s) to train. Use -1 for embedding layer. If not specified, trains all layers.",
    )
    parser.add_argument(
        "--sublayer",
        type=str,
        default="block_output",
        choices=["block_output", "post_attn_resid"],
        help="Where to hook residual stream: block_output (post-layer) or post_attn_resid (after attention, before LayerNorm). i.e. layer output or halfway through layer",
    )
    return parser.parse_args()


def random_subspace_in_pca(
    features: torch.Tensor,
    subspace_dim: int,
    variance_threshold: float = 0.9
) -> torch.Tensor:
    """Generate random orthonormal subspace within top PCs capturing variance_threshold.

    Args:
        features: Activation tensor of shape (n_samples, hidden_dim)
        subspace_dim: Desired dimensionality of the output subspace
        variance_threshold: Fraction of variance to capture (default 0.9)

    Returns:
        Orthonormal matrix of shape (hidden_dim, subspace_dim) representing a random
        subspace within the span of the top PCs. If subspace_dim exceeds the number
        of PCs needed for variance_threshold, pads with random orthogonal directions.
    """
    from sklearn.decomposition import TruncatedSVD

    n_samples, hidden_dim = features.shape

    # Center features (for PCA)
    features_centered = features - features.mean(dim=0, keepdim=True)

    # Compute SVD with enough components to find the variance cutoff
    max_components = min(n_samples - 1, hidden_dim - 1, 500)
    svd = TruncatedSVD(n_components=max_components, algorithm="randomized")
    svd.fit(features_centered.numpy())

    # Find k: smallest number of PCs capturing variance_threshold
    cumvar = np.cumsum(svd.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    k = min(k, max_components)

    # V_k: top k PCs (hidden_dim × k)
    V_k = torch.tensor(svd.components_[:k].T, dtype=features.dtype)

    if subspace_dim <= k:
        # Random rotation within V_k
        G = torch.randn(k, subspace_dim)
        Q, _ = torch.linalg.qr(G)
        return V_k @ Q
    else:
        # Use all k PCs + random orthogonal padding
        G = torch.randn(hidden_dim, subspace_dim - k)
        G_orth = G - V_k @ (V_k.T @ G)  # project out V_k
        Q_extra, _ = torch.linalg.qr(G_orth)
        return torch.cat([V_k, Q_extra], dim=1)

def initialize_subspace_featurizers(targets: dict, subspace_dim: int, init_subspaces={}) -> None:
    """Initialize SubspaceFeaturizers on all units in `targets`

    Args:
        targets: Dict mapping keys to InterchangeTargets
        subspace_dim: Dimension of subspace we want
        init_subspaces: Dict mapping keys to subspace-dim initialization subspaces for each target.
    """
    for _key, target in targets.items():
        init_rotation = init_subspaces.get(_key)
        for unit in target.flatten():
            if unit.shape is None:
                raise ValueError(f"Unit {unit.id} has no shape defined")
            hidden_dim = unit.shape[0]
            if init_rotation is not None:
                # NOTE: if there are multiple model units per target, this will use same rotation for all of them
                unit.set_featurizer(
                    SubspaceFeaturizer(
                        trainable=True,
                        id=f"subspace_{unit.id}",
                        rotation_subspace=init_rotation
                    )
                )
            else:
                unit.set_featurizer(
                    SubspaceFeaturizer(
                        shape=(hidden_dim, subspace_dim),
                        trainable=True,
                        id=f"subspace_{unit.id}",
                    )
                )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize timing tracker
    timings = {}
    total_start = time.time()

    model_name = args.model.split("/")[-1]
    pca_features_path = f"outputs/pca/{model_name}/{args.task_folder}/features"

    # Check if PCA features exist for this sublayer when using PCA-based init
    if args.init in ("pca", "random-pca") and not os.path.exists(pca_features_path):
        raise ValueError(
            f"PCA init requires pre-computed features at {pca_features_path}. "
            f"Run: python scripts/collect_pca.py --task-folder {args.task_folder}"
        )
    dataset_path = os.path.join(
        args.dataset_path,
        model_name,
        args.task_folder,
        "filtered_dataset.json"
    )
    output_path = os.path.join(
        args.output_path,
        model_name,
        args.task_folder
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = ""
    for k, v in vars(args).items():
        config_str += f"{k}{v}"
    hsh = hashlib.md5(config_str.encode()).hexdigest()[:6]

    run_name = f"{timestamp}_{args.target_variable}_{hsh}"
    output_path = os.path.join(output_path, run_name)
    os.makedirs(output_path, exist_ok=True)

    print("=" * 60)
    print(f"DAS Training for Task: {args.task_folder}")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Target variable: {args.target_variable}")
    print(f"Model: {args.model}")
    print(f"Token positions: {args.token_positions}")
    print(f"Subspace dim: {args.subspace_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output dir: {output_path}")
    print("=" * 60)

    # Load dataset from JSON
    print(f"\nLoading dataset from {dataset_path}...")
    t0 = time.time()
    full_dataset = load_dataset(dataset_path)
    total_size = len(full_dataset)
    timings["dataset_loading"] = time.time() - t0
    print(f"  Loaded {total_size} examples ({timings['dataset_loading']:.2f}s)")

    # Split into train/test
    train_size = int(total_size * args.train_split)
    train_data = full_dataset.dataset.select(range(train_size))
    test_data = full_dataset.dataset.select(range(train_size, total_size))

    train_dataset = CounterfactualDataset(dataset=train_data, id=f"{args.task_folder}_train")
    test_dataset = CounterfactualDataset(dataset=test_data, id=f"{args.task_folder}_test")

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")

    # Setup pipeline
    print(f"\nLoading model {args.model}...")
    t0 = time.time()
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=1,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        max_length=64,
    )
    pipeline.tokenizer.padding_side = "left"
    timings["model_loading"] = time.time() - t0
    print(f"  Model loaded ({timings['model_loading']:.2f}s)")

    # Create token positions
    task = re.sub(r'(_no-mtws|_unfiltered)', '', args.task_folder)
    all_token_positions = TASKS[task]["create_token_positions"](pipeline)
    causal_model = TASKS[task]["causal_model"]

    # Select token positions based on argument
    if args.token_positions == ["all"]:
        token_positions = []
        position_names = []
        for k,v in all_token_positions.items():
            token_positions.append(v)
            position_names.append(k)
    else:
        token_positions = [all_token_positions[k] for k in args.token_positions]
        position_names = args.token_positions

    print(f"Using token positions: {position_names}")

    # Save datasets as JSON for train_DAS (load_counterfactual_examples expects a JSON file)
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.json")
        test_path = os.path.join(temp_dir, "test.json")

        with open(train_path, "w") as f:
            json.dump(list(train_dataset.dataset), f)
        with open(test_path, "w") as f:
            json.dump(list(test_dataset.dataset), f)

        # Build interchange targets
        print("\nBuilding interchange targets...")
        t0 = time.time()
        num_layers = pipeline.model.config.num_hidden_layers
        all_layers = [-1] + list(range(num_layers))
        if args.layers is not None:
            layers = [l for l in args.layers if l in all_layers]
            if not layers:
                raise ValueError(f"No valid layers specified. Valid range: -1 to {num_layers - 1}")
        else:
            layers = all_layers

        targets = build_residual_stream_targets(
            pipeline=pipeline,
            layers=layers,
            token_positions=token_positions,
            mode="one_target_per_unit"
        )
        timings["build_targets"] = time.time() - t0
        print(f"  Created {len(targets)} targets across {len(layers)} layers ({timings['build_targets']:.2f}s)")

        # If "pca" or "random-pca" is passed as init method, load in relevant PCA features for each layer/token position.
        # `pca_analysis.py` saved hidden states for the whole dataset, we only want the train examples
        init_subspaces = {}
        if args.init == "pca":
            print("\nLoading PCA initialization...")
            t0 = time.time()
            n_components = args.subspace_dim
            for key, _ in targets.items():  # key = (layer, tok_position)
                samples = load_file(f"{pca_features_path}/{key[0]}__{key[1]}.safetensors")
                inp = {key: samples["features"][:train_size].float()}
                all_components = compute_svd(
                    inp, n_components=n_components, normalize=True  # for it to be PCA
                )[key]["rotation"]  # (model_dim, n_components)
                init_subspaces[key] = all_components
            timings["pca_initialization"] = time.time() - t0
            print(f"  PCA initialization complete ({timings['pca_initialization']:.2f}s)")

        elif args.init == "random-pca":
            print("\nLoading random-PCA initialization (random within top 90% variance PCs)...")
            t0 = time.time()
            n_components = args.subspace_dim
            for key, _ in targets.items():  # key = (layer, tok_position)
                samples = load_file(f"{pca_features_path}/{key[0]}__{key[1]}.safetensors")
                features = samples["features"][:train_size].float()
                init_subspaces[key] = random_subspace_in_pca(features, n_components, variance_threshold=0.9)
            timings["random_pca_initialization"] = time.time() - t0
            print(f"  Random-PCA initialization complete ({timings['random_pca_initialization']:.2f}s)")

        print(f"\nInitializing SubspaceFeaturizers with {args.subspace_dim} dimensions...")
        initialize_subspace_featurizers(targets, args.subspace_dim, init_subspaces)

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Train DAS with pre-initialized featurizers
        print(f"\nTraining DAS...")
        t0 = time.time()
        result = train_DAS(
            interchange_targets=targets,
            causal_model=causal_model,
            train_dataset_path=train_path,
            test_dataset_path=test_path,
            pipeline=pipeline,
            target_variable_group=(args.target_variable,),
            output_dir=output_path,
            metric=metric,
            verbose=True,
            config={
                "intervention_type": "interchange",
                "train_batch_size": args.batch_size,
                "evaluation_batch_size": args.batch_size,
                "training_epoch": args.epochs,
                "init_lr": args.lr
            },
        )
        timings["training"] = time.time() - t0
        print(f"  Training complete ({timings['training']:.2f}s)")

    # Compute total time
    timings["total"] = time.time() - total_start

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best training score: {result['metadata']['train_max_score']:.3f}")
    print(f"Best test score: {result['metadata']['test_max_score']:.3f}")
    print(f"Best train location: {result['metadata']['train_best_location']}")
    print(f"Best test location: {result['metadata']['test_best_location']}")
    print(f"\nResults saved to: {output_path}")

    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for phase, duration in timings.items():
        if phase != "total":
            print(f"  {phase}: {duration:.2f}s")
    print("-" * 60)
    print(f"  Total: {timings['total']:.2f}s")


if __name__ == "__main__":
    main()
