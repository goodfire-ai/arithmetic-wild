"""
PCA + Least-Squares Fourier Probes Sweep — trains cos/sin probes on
PCA-reduced activations across layers of Llama-3.1-8B.

Based on the method described in the paper "NOT ALL LANGUAGE MODEL FEATURES ARE
ONE-DIMENSIONALLY LINEAR" [https://arxiv.org/pdf/2405.14860]

Method: PCA(dim=5) on activations → least-squares fit of cos/sin targets.
Supports 4 tasks: addition, weekdays, months, hours.
Supports 3 activation sources: resid (residual stream), mlp_input, mlp_output.

Usage:
    python pca_fourier_probes_sweep.py --task weekdays --layer 18 --position last_token --source resid
    python pca_fourier_probes_sweep.py --task addition --layer 18 --position last_token --source mlp_input --variable input,offset
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B"
# Default output folder is <repo>/outputs/pca_fourier_probes. The script lives
# at <repo>/src/fourier_probes/, so parents[2] resolves to the repo root.
SAVE_BASE = str(Path(__file__).resolve().parents[2] / "outputs" / "pca_fourier_probes")
BATCH_SIZE = 64
DEFAULT_PCA_DIM = 5


# --- Summation ---
SUM_A_VALUES = list(range(1, 200))
SUM_B_VALUES = list(range(1, 200))
SUM_TEMPLATE = "{input}+{offset}="

# --- Weekdays ---
DAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_TO_INDEX = {d: i for i, d in enumerate(DAYS)}
DAY_OFFSETS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
               "thirteen", "fourteen"]
DAY_OFFSET_TO_NUM = {w: i + 1 for i, w in enumerate(DAY_OFFSETS)}
WEEKDAY_TEMPLATE = "Q: What day is {offset} days after {input}?\nA:"

# --- Months ---
MONTHS = ["December", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November"]
MONTH_TO_INDEX = {m: i for i, m in enumerate(MONTHS)}
MONTH_OFFSETS = ["one", "two", "three", "four", "five", "six",
               "seven", "eight", "nine", "ten", "eleven", "twelve",
               "thirteen", "fourteen", "fifteen", "sixteen",
               "seventeen", "eighteen", "nineteen", "twenty",
               "twenty-one", "twenty-two", "twenty-three", "twenty-four"]
MONTH_OFFSET_TO_NUM = {w: i + 1 for i, w in enumerate(MONTH_OFFSETS)}
MONTH_TEMPLATE = "Q: What month is {offset} months after {input}?\nA:"

# --- Hours ---
HOURS = [f"{h:02d}" for h in range(24)]
HOUR_TO_INT = {h: int(h) for h in HOURS}
HOUR_OFFSETS = [
    "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
    "twenty-one", "twenty-two", "twenty-three", "twenty-four",
    "twenty-five", "twenty-six", "twenty-seven", "twenty-eight",
    "twenty-nine", "thirty", "thirty-one", "thirty-two",
    "thirty-three", "thirty-four", "thirty-five", "thirty-six",
    "thirty-seven", "thirty-eight", "thirty-nine", "forty",
    "forty-one", "forty-two", "forty-three", "forty-four",
    "forty-five", "forty-six", "forty-seven", "forty-eight"
]
HOUR_OFFSET_TO_NUM = {w: i + 1 for i, w in enumerate(HOUR_OFFSETS)}
HOUR_TEMPLATE = ("Q: In 24-hour time, it is now {input}:00. "
                 "What time will it be in {offset} hours?\n"
                 "A: In 24-hour time, it will be ")

# ── Task configurations ─────────────────────────────────────────────────────

TASK_CONFIGS = {
    "addition": {
        "natural_period": None,  # sweep over periods
        "variables": {
            "input": lambda s: int(s["input"]),
            "offset": lambda s: int(s["offset"]),
            "output": lambda s: int(s["input"]) + int(s["offset"]),
        },
        "positions": ["input", "offset", "last_token"],
    },
    "weekdays": {
        "natural_period": 7,
        "variables": {
            "offset": lambda s: DAY_OFFSET_TO_NUM[s["offset"]],
            "input": lambda s: DAY_TO_INDEX[s["input"]],
            "output": lambda s: DAY_TO_INDEX[s["output"]],
        },
        "positions": ["offset", "input", "last_token"],
    },
    "months": {
        "natural_period": 12,
        "variables": {
            "offset": lambda s: MONTH_OFFSET_TO_NUM[s["offset"]],
            "input": lambda s: MONTH_TO_INDEX[s["input"]],
            "output": lambda s: MONTH_TO_INDEX[s["output"]],
        },
        "positions": ["offset", "input", "last_token"],
    },
    "hours": {
        "natural_period": [24],
        "variables": {
            "offset": lambda s: HOUR_OFFSET_TO_NUM[s["offset"]],
            "input": lambda s: HOUR_TO_INT[s["input"]],
            "output": lambda s: HOUR_TO_INT[s["output"]],
        },
        "positions": ["offset", "input", "last_token"],
    },
}


def generate_samples(task_name):
    """Generate all input combinations for a task, return list of dicts with variable values."""
    if task_name == "addition":
        samples = []
        for a in SUM_A_VALUES:
            for b in SUM_B_VALUES:
                samples.append({
                    "input": a,
                    "offset": b,
                    "output": a + b,
                    "raw_input": SUM_TEMPLATE.format(input=a, offset=b),
                    "expected": str(a + b),
                })
        return samples

    elif task_name == "weekdays":
        samples = []
        for day in DAYS:
            for offset in DAY_OFFSETS:
                day_idx = DAY_TO_INDEX[day]
                offset_num = DAY_OFFSET_TO_NUM[offset]
                result = DAYS[(day_idx + offset_num) % 7]
                samples.append({
                    "input": day,
                    "offset": offset,
                    "output": result,
                    "raw_input": WEEKDAY_TEMPLATE.format(offset=offset, input=day),
                    "expected": " " + result,
                })
        return samples

    elif task_name == "months":
        samples = []
        for month in MONTHS:
            for offset in MONTH_OFFSETS:
                month_idx = MONTH_TO_INDEX[month]
                offset_num = MONTH_OFFSET_TO_NUM[offset]
                result = MONTHS[(month_idx + offset_num) % 12]
                samples.append({
                    "input": month,
                    "offset": offset,
                    "output": result,
                    "raw_input": MONTH_TEMPLATE.format(offset=offset, input=month),
                    "expected": " " + result,
                })
        return samples

    elif task_name == "hours":
        samples = []
        for hour in HOURS:
            for offset in HOUR_OFFSETS:
                h = HOUR_TO_INT[hour]
                offset_num = HOUR_OFFSET_TO_NUM[offset]
                result = f"{(h + offset_num) % 24:02d}"
                samples.append({
                    "input": hour,
                    "offset": offset,
                    "output": result,
                    "raw_input": HOUR_TEMPLATE.format(input=hour, offset=offset),
                    "expected": result,
                })
        return samples


def filter_correct_samples(model, tokenizer, samples, task_name, device):
    """Run the model on all prompts and keep only samples where the model's
    top-1 prediction matches the expected output's first token."""
    prompts = [s["raw_input"] for s in samples]
    expected = [s["expected"] for s in samples]

    # Tokenize expected outputs to get the first expected token
    expected_token_ids = []
    for exp in expected:
        toks = tokenizer.encode(exp, add_special_tokens=False)
        expected_token_ids.append(toks[0])

    correct_mask = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            logits = model(**inputs).logits
            # Get top-1 prediction at last token
            pred_ids = logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            for j, pred_id in enumerate(pred_ids):
                correct_mask.append(pred_id == expected_token_ids[i + j])

    filtered = [s for s, c in zip(samples, correct_mask) if c]
    n_correct = sum(correct_mask)
    print(f"Accuracy: {n_correct}/{len(samples)} ({100 * n_correct / len(samples):.1f}%)")
    return filtered


TEMPLATES = {
    "addition": SUM_TEMPLATE,
    "weekdays": WEEKDAY_TEMPLATE,
    "months": MONTH_TEMPLATE,
    "hours": HOUR_TEMPLATE,
}


def find_variable_token_position(tokenizer, sample, task_name, position):
    """Token index of the last token of `position`'s span in the formatted prompt.

    Returns -1 for 'last_token' (handled specially by the caller).
    """
    if position == "last_token":
        return -1
    template = TEMPLATES.get(task_name)
    if template is None:
        return -1
    placeholder = "{" + position + "}"
    idx = template.find(placeholder)
    if idx < 0:
        return -1
    prefix = template[: idx + len(placeholder)].format(**sample)
    return len(tokenizer.encode(prefix, add_special_tokens=False)) - 1


def collect_activations(model, tokenizer, samples, layer, position, task_name, device, source="resid"):
    """Collect activations at a given layer, position, and source.

    Args:
        source: "resid" (residual stream output of layer),
                "mlp_input" (input to MLP),
                "mlp_output" (output of MLP)
    """
    prompts = [s["raw_input"] for s in samples]
    collected = []
    token_positions = None

    if position != "last_token":
        token_positions = [
            find_variable_token_position(tokenizer, s, task_name, position)
            for s in samples
        ]

    # Precompute per-prompt token lengths (un-padded) for padding offset correction
    prompt_token_lengths = None
    if token_positions is not None:
        prompt_token_lengths = [
            len(tokenizer.encode(p, add_special_tokens=False))
            for p in prompts
        ]

    batch_start_idx = 0

    def extract_at_positions(hidden):
        if token_positions is None:
            collected.append(hidden[:, -1, :].detach().float().cpu())
            return
        bs, seq_len = hidden.shape[:2]
        # With left padding, un-padded position p maps to (seq_len - prompt_len + p).
        pos = torch.as_tensor(
            token_positions[batch_start_idx:batch_start_idx + bs], device=hidden.device
        )
        plen = torch.as_tensor(
            prompt_token_lengths[batch_start_idx:batch_start_idx + bs], device=hidden.device
        )
        adjusted = seq_len - plen + pos
        acts = hidden[torch.arange(bs, device=hidden.device), adjusted]
        collected.append(acts.detach().float().cpu())

    if source == "resid":
        def hook_fn(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            extract_at_positions(hidden)
        hook = model.model.layers[layer].register_forward_hook(hook_fn)
    elif source == "mlp_input":
        def hook_fn(module, inp, out):
            hidden = inp[0] if isinstance(inp, tuple) else inp
            extract_at_positions(hidden)
        hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    elif source == "mlp_output":
        def hook_fn(module, inp, out):
            hidden = out if not isinstance(out, tuple) else out[0]
            extract_at_positions(hidden)
        hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown source: {source}")

    with torch.no_grad():
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_start_idx = i
            batch = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            model(**inputs)
            if (i // BATCH_SIZE) % 50 == 0:
                print(f"  batch {i // BATCH_SIZE}/{(len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE}")

    hook.remove()
    return torch.cat(collected, dim=0)


def fit_fourier_probe(pca_acts, values, period, k=1, test_size=0.0, random_state=42):
    """Fit cos/sin probe via least squares on PCA'd activations.

    Args:
        pca_acts: (N, pca_dim) tensor of PCA'd activations
        values: (N,) tensor of integer values
        period: the period p for 2*pi*k/p
        k: frequency index (default 1)
        test_size: fraction of data to hold out for evaluation (0 = no split)
        random_state: random seed for train/test split

    Returns:
        dict with probe_q, probe_r, target_to_embedding, r2_cos, r2_sin, r2_avg
    """
    w = 2 * np.pi * k / period

    # Build cos/sin targets
    cos_targets = torch.cos(w * values.float())
    sin_targets = torch.sin(w * values.float())
    multid_targets = torch.stack([cos_targets, sin_targets], dim=1)  # (N, 2)

    # Train/test split
    if test_size > 0:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(pca_acts))
        idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
        X_train, X_test = pca_acts[idx_train], pca_acts[idx_test]
        y_train, y_test = multid_targets[idx_train], multid_targets[idx_test]
    else:
        X_train = pca_acts
        y_train = multid_targets
        X_test = pca_acts
        y_test = multid_targets

    # Least squares fit on training data
    solution = torch.linalg.lstsq(X_train, y_train).solution  # (pca_dim, 2)

    # Evaluate on test data
    predictions = X_test @ solution

    # QR decomposition for probe storage
    probe_q, probe_r = torch.linalg.qr(solution)

    # Build target_to_embedding: maps each discrete value to its [cos, sin] embedding
    all_vals = torch.arange(period)
    target_to_embedding = torch.stack([
        torch.cos(w * all_vals.float()),
        torch.sin(w * all_vals.float()),
    ], dim=1)  # (period, 2)

    # R² per component (on test data)
    r2s = []
    for dim in range(2):
        y = y_test[:, dim]
        y_pred = predictions[:, dim]
        ss_res = ((y - y_pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")
        r2s.append(r2)

    return {
        "probe_q": probe_q,
        "probe_r": probe_r,
        "target_to_embedding": target_to_embedding,
        "r2_cos": r2s[0],
        "r2_sin": r2s[1],
        "r2_avg": np.mean(r2s),
    }


def main():
    parser = argparse.ArgumentParser(description="PCA + Least-Squares Fourier Probes Sweep")
    parser.add_argument("--task", type=str, required=True,
                        choices=["addition", "weekdays", "months", "hours"],
                        help="Task to probe")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer index (0-31)")
    parser.add_argument("--position", type=str, default="last_token",
                        help="Token position (last_token, input, or offset)")
    parser.add_argument("--source", type=str, default="resid",
                        choices=["resid", "mlp_input", "mlp_output"],
                        help="Activation source (default: resid)")
    parser.add_argument("--variable", type=str, default=None,
                        help="Comma-separated list of variables to probe (default: all for the task)")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                        help=f"PCA components (default {DEFAULT_PCA_DIM})")
    parser.add_argument("--max-period", type=int, default=150,
                        help="Max period for addition sweep (default 150)")
    parser.add_argument("--periods", type=str, default=None,
                        help="Comma-separated list of specific periods to probe (overrides natural_period and sweep)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for held-out R² evaluation (default 0.2, set 0 to disable)")
    parser.add_argument("--save-base", type=str, default=SAVE_BASE,
                        help=f"Output directory (default: {SAVE_BASE})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_cfg = TASK_CONFIGS[args.task]

    print(f"=== PCA Fourier Probe Sweep ===")
    print(f"Task: {args.task}, Layer: {args.layer}, Position: {args.position}, Source: {args.source}")
    print(f"PCA dim: {args.pca_dim}, Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loaded, hidden_size={model.config.hidden_size}")

    # ── Generate samples and filter to correct ones ─────────────────────
    samples = generate_samples(args.task)
    print(f"Total prompts: {len(samples)}")
    print("Filtering to prompts the model gets correct...")
    samples = filter_correct_samples(model, tokenizer, samples, args.task, device)
    print(f"Prompts after filtering: {len(samples)}")

    # ── Collect activations ──────────────────────────────────────────────
    print(f"Collecting {args.source} activations at layer {args.layer}, position={args.position}...")
    activations = collect_activations(
        model, tokenizer, samples, args.layer, args.position, args.task, device,
        source=args.source,
    )
    print(f"Activations shape: {activations.shape}")

    # ── PCA ───────────────────────────────────────────────────────────────
    print(f"Fitting PCA(n_components={args.pca_dim})...")
    pca = PCA(n_components=args.pca_dim)
    pca_acts = torch.tensor(pca.fit_transform(activations.numpy()), dtype=torch.float32)
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # ── Select variables to probe ────────────────────────────────────────
    if args.variable:
        var_names = [v.strip() for v in args.variable.split(",")]
        variables = {k: v for k, v in task_cfg["variables"].items() if k in var_names}
        if not variables:
            raise ValueError(f"No matching variables found. Available: {list(task_cfg['variables'].keys())}")
    else:
        variables = task_cfg["variables"]

    # ── Fit probes per variable ──────────────────────────────────────────
    for var_name, value_fn in variables.items():
        values = torch.tensor([value_fn(s) for s in samples])

        if args.periods:
            periods = [int(m) for m in args.periods.split(",")]
        elif task_cfg["natural_period"] is not None:
            nm = task_cfg["natural_period"]
            periods = nm if isinstance(nm, list) else [nm]
        else:
            max_val = values.max().item()
            max_period = min(max_val // 2, args.max_period)
            periods = list(range(2, max_period + 1))

        save_dir = os.path.join(
            args.save_base, args.task, var_name,
            f"layer_{args.layer}", f"pos_{args.position}", args.source
        )
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Variable: {var_name}, periods: {periods[0]}..{periods[-1]} ({len(periods)} values)")
        print(f"Saving to: {save_dir}")

        all_r2 = []

        for i, m in enumerate(periods):
            result = fit_fourier_probe(pca_acts, values, period=m, k=1, test_size=args.test_size)

            # Save probe
            torch.save({
                "layer": args.layer,
                "position": args.position,
                "source": args.source,
                "task": args.task,
                "variable": var_name,
                "period": m,
                "k": 1,
                "pca_dim": args.pca_dim,
                "probe_q": result["probe_q"],
                "probe_r": result["probe_r"],
                "target_to_embedding": result["target_to_embedding"],
                "pca_components": torch.tensor(pca.components_, dtype=torch.float32),
                "pca_mean": torch.tensor(pca.mean_, dtype=torch.float32),
            }, os.path.join(save_dir, f"probe_period{m}.pt"))

            all_r2.append({
                "period": int(m),
                "r2_cos": float(result["r2_cos"]) if not np.isnan(result["r2_cos"]) else None,
                "r2_sin": float(result["r2_sin"]) if not np.isnan(result["r2_sin"]) else None,
                "r2_avg": float(result["r2_avg"]) if not np.isnan(result["r2_avg"]) else None,
            })

            if (i + 1) % 20 == 0 or i == len(periods) - 1:
                print(f"  [{i + 1:3d}/{len(periods)}] period {m:3d}  "
                      f"R2_cos={result['r2_cos']:.4f}  R2_sin={result['r2_sin']:.4f}  "
                      f"R2_avg={result['r2_avg']:.4f}")

        # Save R² summary
        r2_save = {
            "task": args.task,
            "variable": var_name,
            "layer": int(args.layer),
            "position": args.position,
            "source": args.source,
            "pca_dim": int(args.pca_dim),
            "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "pca_total_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "n_samples": int(len(samples)),
            "n_samples_correct_only": True,
            "test_size": args.test_size,
            "results": all_r2,
        }
        with open(os.path.join(save_dir, "r2_results.json"), "w") as f:
            json.dump(r2_save, f, indent=2)

    print(f"\nDone! Probes saved under {args.save_base}/{args.task}/")


if __name__ == "__main__":
    main()
