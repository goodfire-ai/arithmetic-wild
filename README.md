# Arithmetic in the Wild: Llama uses Base-10 Addition to Reason about Cyclic Concepts

Companion code for [Arithmetic in the Wild: Llama Uses Base-10 Addition to Reason about Cyclic Concepts](https://arxiv.org/abs/2605.01148). If you have questions, please email `sheridan.feucht[at]gmail.com` and `tal[at]goodfire.ai`.

Also see our [addition neuron explorer](https://sfeucht.github.io/addition-neuron-explorer/) to look at neuron activations across all tasks, projected onto Fourier planes. 

## Table of Contents

### Section 2: DAS Code 
- `load_das_subspaces.ipynb` gives a code snippet for loading in the exact trained DAS subspaces we used in the paper from Hugging Face Hub.
- `train_das.py` is the code we used to train DAS subspaces. If you want to initialize subspaces with PCA directions, first run `python src/collect_pca.py`. 

### Section 3: Cross-Task Patching
- `cross_task_patch.ipynb` gives a lightweight example of patching from `addition` to `months`. 
- `cross_task_patch.py` applies this procedure across every possible combination of source and target tasks.

### Section 4: Fourier Probes 

All under `src/fourier_probes/`.

- `fourier_probes_sweep.py` trains Fourier probes (linear directions encoding cosine and sine waves) for a range of $T$. Saves separate `probe_period{T}_{cos,sin}.pt` files.
- `pca_fourier_probes_sweep.py` trains the same probes on PCA-reduced activations (PCA dim=5 by default) via least-squares fit. Supports addition, weekdays, months, and hours; sweeps periods per task and reports held-out R². Saves a single `probe_period{T}.pt` per period.
- `fourier_steering.ipynb` uses these probes trained on addition to steer on cyclic tasks.
- `addition_probes_layer_18_last_token_resid/` pretrained Fourier probes (periodicties 2, 5, 10, 20, 50, 100; cos+sin) for the addition task at layer 18, last-token, residual stream.

### Section 5: Neurons 
- `neuron_selection.py` finds addition neurons by looking at their overlap with DAS subspaces. 
- `neurons_across_sums.ipynb` gives code for reproducing Figure 8 (neuron activations across output sums).
- `neurons_across_prompts.ipynb` gives plotting code to get gate, up, and full activation visualizations for different tasks.
- `static_downproj.ipynb` shows down projection vectors on Fourier planes. 
- `dynamic_downproj.ipynb` shows down projections scaled by their activations for an actual forward pass on Fourier planes. 

## Setup

Requires Python >=3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo-url> arithmetic-wild
cd arithmetic-wild
uv sync
```

This creates `.venv/` and installs all dependencies (including `causalab` from the pinned git revision in `pyproject.toml`) using the locked versions in `uv.lock`.

`torch` is pinned to the CUDA 12.8 build, so you need an NVIDIA driver that supports CUDA ≥ 12.8 (run `nvidia-smi` to check). If yours is older, swap `cu128` for a matching version (e.g. `cu126`) in `pyproject.toml` and re-run `uv lock && uv sync`.

Run scripts with:
```bash
uv run python scripts/your_script.py
```

Or activate the venv directly: `source .venv/bin/activate`.

To bump the pinned `causalab` revision, edit `rev` under `[tool.uv.sources]` in `pyproject.toml` and run `uv lock` (then commit both files).