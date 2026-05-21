"""Fourier Probes Sweep

The probe checks whether a model's hidden state at a chosen layer / token
position represents an integer variable in a Fourier form: for
each period ``T`` and each integer-valued variable ``n``, we fit a linear
probe to predict ``sin(2*pi*n/T)`` and ``cos(2*pi*n/T)`` and report R² on a
held-out split.

Example:
    python fourier_probes_sweep.py \\
        --layer 18 --position last_token \\
        --source resid --max-period 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from causalab.neural.pipeline import LMPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Defaults ────────────────────────────────────────────────────────────────
TASK_NAME = "addition"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "fourier_probes"
DEFAULT_BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Addition task definitions  ─────────────
INPUT_VALUES = [str(i) for i in range(1, 201)]   # "1".."200"
OFFSET_VALUES = [str(i) for i in range(1, 201)]  # "1".."200"
TEMPLATE = "{a}+{b}="


def inp_to_idx(s: str) -> int:
    return int(s)


def num_to_idx(s: str) -> int:
    return int(s)


def make_prompt(a: str, b: str) -> str:
    return TEMPLATE.format(a=a, b=b)


class _TokenIndex:
    """Tiny stand-in for causalab's TokenPosition: holds a per-prompt index
    function and exposes ``.index(sample) -> [int]`` like the original."""

    def __init__(self, fn):
        self._fn = fn

    def index(self, sample):
        return self._fn(sample)


def addition_token_positions(pipeline: "LMPipeline") -> dict[str, _TokenIndex]:
    """For prompts of the form ``{a}+{b}=``, return token-index lookups for:
    - ``input``      — last token of ``a`` (left of ``+``)
    - ``offset``     — last token of ``b`` (between ``+`` and ``=``)
    - ``last_token`` — final token of the whole prompt

    Indices are computed against ``tokenizer.encode(...)`` (default
    ``add_special_tokens=True``), so they include the BOS — which matches
    what ``tokenizer(p)["input_ids"]`` produces in ``collect_activations``.
    """
    tokenizer = pipeline.tokenizer

    def _input_end(sample):
        a_str = sample["raw_input"].split("+")[0]
        return [len(tokenizer.encode(a_str)) - 1]

    def _offset_end(sample):
        prefix = sample["raw_input"].split("=")[0]
        return [len(tokenizer.encode(prefix)) - 1]

    def _last(sample):
        return [len(tokenizer.encode(sample["raw_input"])) - 1]

    return {
        "input":      _TokenIndex(_input_end),
        "offset":     _TokenIndex(_offset_end),
        "last_token": _TokenIndex(_last),
    }


# ── Probe ────────────────────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_probe(
    probe: LinearProbe,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    lr: float = LR,
    epochs: int = EPOCHS,
) -> float:
    if y_tr.std() < 1e-6:
        return float("nan")
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        loss = loss_fn(probe(X_tr), y_tr)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(X_te)
        ss_res = ((y_te - pred) ** 2).sum().item()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")
    return r2


# ── Data ────────────────────────────────────────────────────────────────────
def enumerate_prompts() -> tuple[list[str], list[str], list[str]]:
    """Cartesian product of INPUT_VALUES × OFFSET_VALUES, formatted by TEMPLATE."""
    prompts, inp_values, num_values = [], [], []
    for a in INPUT_VALUES:
        for b in OFFSET_VALUES:
            prompts.append(make_prompt(a, b))
            inp_values.append(a)
            num_values.append(b)
    return prompts, inp_values, num_values


def integer_variables(
    inp_values: list[str], num_values: list[str]
) -> dict[str, np.ndarray]:
    """Per-prompt integer values for each variable: input, offset, output."""
    inp_int = np.array([inp_to_idx(a) for a in inp_values])
    num_int = np.array([num_to_idx(b) for b in num_values])
    out_int = inp_int + num_int  # addition: output is the sum

    return {"input": inp_int, "offset": num_int, "output": out_int}


# ── Activation collection ───────────────────────────────────────────────────
def collect_activations(
    pipeline: LMPipeline,
    prompts: list[str],
    layer: int,
    source: str,
    token_position,  # TokenPosition
    batch_size: int,
    inp_values: list | None = None,
    num_values: list | None = None,
) -> np.ndarray:
    """Run prompts through the model, saving one hidden vector per prompt
    at the chosen layer/source/token position. ``inp_values``/``num_values``
    are required when the token position is variable-scoped (e.g. ``input``
    or ``offset``) so the indexer can locate the variable's tokens; for
    ``last_token`` they're optional."""
    model = pipeline.model
    tokenizer = pipeline.tokenizer

    if inp_values is not None and num_values is not None:
        idx_inputs = [
            {"raw_input": p, "input": iv, "offset": nv}
            for p, iv, nv in zip(prompts, inp_values, num_values)
        ]
    else:
        idx_inputs = [{"raw_input": p} for p in prompts]
    unpadded_idx = [token_position.index(s)[0] for s in idx_inputs]
    prompt_lens = [len(tokenizer(p)["input_ids"]) for p in prompts]

    collected: list[torch.Tensor] = []
    batch_start = 0

    def hook_fn(module, inp, out):
        if source == "resid":
            hidden = out[0] if isinstance(out, tuple) else out
        elif source == "mlp_input":
            hidden = inp[0] if isinstance(inp, tuple) else inp
        else:  # mlp_output
            hidden = out[0] if isinstance(out, tuple) else out

        bsz = hidden.shape[0]
        seq_len = hidden.shape[1]
        idx_batch = unpadded_idx[batch_start:batch_start + bsz]
        len_batch = prompt_lens[batch_start:batch_start + bsz]
        acts = torch.stack([
            hidden[j, seq_len - len_batch[j] + idx_batch[j], :]
            for j in range(bsz)
        ])
        collected.append(acts.detach().float().cpu())

    if source == "resid":
        hook = model.model.layers[layer].register_forward_hook(hook_fn)
    else:
        hook = model.model.layers[layer].mlp.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            n_batches = (len(prompts) + batch_size - 1) // batch_size
            for i in range(0, len(prompts), batch_size):
                batch_start = i
                batch = prompts[i:i + batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True
                ).to(model.device)
                model(**inputs)
                if (i // batch_size) % 200 == 0:
                    print(f"  batch {i // batch_size}/{n_batches}")
    finally:
        hook.remove()

    return torch.cat(collected, dim=0).numpy()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=f"Fourier Probes Sweep ({TASK_NAME} task)"
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer index")
    parser.add_argument("--position", required=True,
                        help="Token position name (one of: input, offset, last_token)")
    parser.add_argument("--source", required=True,
                        choices=["mlp_input", "mlp_output", "resid"],
                        help="Activation site: mlp_input, mlp_output, or resid "
                             "(full residual stream, i.e. layer output)")
    parser.add_argument("--variable", default=None,
                        help="Integer variable to probe: input, offset, or output. "
                             "Default: probe all three.")
    parser.add_argument("--max-period", type=int, default=150,
                        help="Maximum period to sweep (default: 150).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HF model name or path (default: {DEFAULT_MODEL}).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Where to write probes + r2_results.json "
                             "(default: <repo>/outputs/fourier_probes).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    print("=== Fourier Probe Sweep ===")
    print(f"Task: {TASK_NAME}, Layer: {args.layer}, "
          f"Position: {args.position}, Source: {args.source}")
    print(f"Model: {args.model}, Max period: {args.max_period}")

    # ── Load model via LMPipeline (handles tokenizer + device + dtype) ───
    pipeline = LMPipeline(
        args.model,
        padding_side="left",
        dtype=torch.bfloat16,
    )
    pipeline.model.eval()
    hidden_size = pipeline.model.config.hidden_size
    print(f"Loaded {args.model}, hidden_size={hidden_size}, "
          f"device={pipeline.model.device}")

    # ── Resolve token position ───────────────────────────────────────────
    token_positions = addition_token_positions(pipeline)
    if args.position not in token_positions:
        raise SystemExit(
            f"Unknown --position {args.position!r}. "
            f"Available: {sorted(token_positions)}"
        )
    tp = token_positions[args.position]

    # ── Build prompts + per-prompt integer values ────────────────────────
    prompts, inp_values, num_values = enumerate_prompts()
    int_vars = integer_variables(inp_values, num_values)
    print(f"Total prompts: {len(prompts)}")
    print(f"Integer-mappable variables: {sorted(int_vars)}")

    if args.variable is not None and args.variable not in int_vars:
        raise SystemExit(
            f"--variable {args.variable!r} is not integer-mappable. "
            f"Available: {sorted(int_vars)}"
        )

    # ── Collect activations ──────────────────────────────────────────────
    activations = collect_activations(
        pipeline=pipeline,
        prompts=prompts,
        layer=args.layer,
        source=args.source,
        token_position=tp,
        batch_size=args.batch_size,
        inp_values=inp_values,
        num_values=num_values,
    )
    print(f"Activations shape: {activations.shape}")

    # ── Train/test split ─────────────────────────────────────────────────
    X_all = torch.tensor(activations, dtype=torch.float32)
    input_dim = X_all.shape[1]
    X_train, X_test, idx_train, idx_test = train_test_split(
        X_all, np.arange(len(X_all)),
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    # ── Pick which variable(s) to probe ──────────────────────────────────
    variables = (
        {args.variable: int_vars[args.variable]}
        if args.variable is not None
        else int_vars
    )

    output_root = Path(args.output_dir).expanduser().resolve()

    for var_name, var_values in variables.items():
        max_period = min(int(var_values.max()) // 2, args.max_period)
        periods = np.arange(2, max_period + 1)

        var_train = var_values[idx_train]
        var_test = var_values[idx_test]

        save_dir = (
            output_root
            / TASK_NAME
            / args.source
            / var_name
            / f"layer_{args.layer}"
            / f"pos_{args.position}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Variable: {var_name} (range {int(var_values.min())}.."
              f"{int(var_values.max())}, max_period={max_period})")
        print(f"Saving to: {save_dir}")
        print(f"{'=' * 60}")

        r2_sin: list[float] = []
        r2_cos: list[float] = []

        for i, m in enumerate(periods):
            angle_train = 2 * np.pi * var_train / m
            angle_test = 2 * np.pi * var_test / m

            for func_name, func in [("sin", np.sin), ("cos", np.cos)]:
                y_tr = torch.tensor(func(angle_train), dtype=torch.float32)
                y_te = torch.tensor(func(angle_test), dtype=torch.float32)

                lp = LinearProbe(input_dim)
                r2 = train_probe(lp, X_train, y_tr, X_test, y_te)

                (r2_sin if func_name == "sin" else r2_cos).append(r2)

                torch.save(
                    lp.state_dict(),
                    save_dir / f"probe_period{m}_{func_name}.pt",
                )

            if (i + 1) % 20 == 0 or i == len(periods) - 1:
                sin_str = f"{r2_sin[-1]:.4f}" if not np.isnan(r2_sin[-1]) else "  skip"
                cos_str = f"{r2_cos[-1]:.4f}" if not np.isnan(r2_cos[-1]) else "  skip"
                print(f"  [{i + 1:3d}/{len(periods)}] period {m:3d}  "
                      f"R²_sin={sin_str}  R²_cos={cos_str}")

        r2_save = {
            "task": TASK_NAME,
            "variable": var_name,
            "layer": int(args.layer),
            "position": args.position,
            "activation_source": args.source,
            "model": args.model,
            "periods": [int(m) for m in periods],
            "r2_sin": [float(x) if not np.isnan(x) else None for x in r2_sin],
            "r2_cos": [float(x) if not np.isnan(x) else None for x in r2_cos],
            "input_dim": int(input_dim),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "max_period": int(max_period),
        }
        with open(save_dir / "r2_results.json", "w") as f:
            json.dump(r2_save, f, indent=2)

    print(f"\nDone! Probes saved under {output_root}/{TASK_NAME}/{args.source}/")


if __name__ == "__main__":
    main()
