import re
import os 
import json
from typing import Dict, Any
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_from_disk, Dataset
from causalab.neural.featurizers import Featurizer
from cl_patch import CounterfactualDataset

# for DAS, we go up to three cycles, and then filter out incorrect prompts. 
THREE_CYCLES = {
    "weekdays" : 21,
    "months" : 36,
    "hours" : 71
}

# in some experiments we restrict to two cycles to cleanly exclude incorrect prompts. 
TWO_CYCLES = {
    "weekdays" : 14,
    "months" : 24,
    "hours" : 47
}

# Check if neural network output matches causal model output.
def metric(neural_output: Dict[str, Any], causal_output: str) -> bool:
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    # For numeric outputs, extract the leading number from the neural output
    # to handle continuation tokens (e.g., "10, 4+" -> "10") while still
    # avoiding substring false positives (e.g., "7" in "17")
    if causal_str.lstrip("-").isdigit():
        m = re.match(r"-?\d+", neural_str)
        return m is not None and m.group() == causal_str
    return causal_str in neural_str or neural_str in causal_str

def load_dataset(dataset_path: str) -> CounterfactualDataset:
    """Load dataset from either JSON or HuggingFace disk format.

    Args:
        dataset_path: Path to dataset (JSON file or HuggingFace disk directory)

    Returns:
        CounterfactualDataset
    """
    path = Path(dataset_path)

    # Check if it's a JSON file
    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return CounterfactualDataset.from_dict(data, id=path.stem)

    # Otherwise, try loading as HuggingFace disk format
    hf_dataset = load_from_disk(dataset_path)
    if not isinstance(hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(hf_dataset).__name__}")
    return CounterfactualDataset(dataset=hf_dataset, id=path.name)

# Tal535/Arithmetic-In-The-Wild_DAS

def load_subspace_hf(task, variable, layer=18, token_position="last_token"):
    assert layer == 18 and token_position == "last_token"
    assert task in ["addition", "hours", "months", "weekdays"]
    assert variable in ["input", "output", "offset"]

    before_mlp = True if variable in ["input", "offset"] else False 

    # o for MLP output, s for sublayer right before MLP
    folder_name = f"{task}_{variable}_l{layer}"
    folder_name += "s" if before_mlp else "o" 
    
    # returns absolute path to the cached file
    local_dir = snapshot_download(
        repo_id="Tal535/Arithmetic-In-The-Wild_DAS",
        repo_type="dataset",
        allow_patterns=folder_name + "/*",   # or ["path/in/**"] for recursive
    )

    featurizer = Featurizer.load_modules(local_dir + "/" + folder_name + "/model")
    subspace = featurizer.featurizer.rotate.weight.detach().cuda()  # (model_dim, subspace_dim)
    print(f"  Subspace shape: {subspace.shape}")
    return subspace 


def load_subspace_old(run_id, task_folder, subspace_layer, token_position="last_token", is_sublayer=False, model_name="Llama-3.1-8B"):
    if subspace_layer == -1:
        modulename = f"ResidualStream(Layer-0,block_input,Token-{token_position})"
    else:
        if is_sublayer:
            # e.g. ResidualStream(Layer-18,post_attn_resid,Token-last_token)
            modulename = f"ResidualStream(Layer-{subspace_layer},post_attn_resid,Token-{token_position})"
        else:
            modulename = f"ResidualStream(Layer-{subspace_layer},block_output,Token-{token_position})"

    das_path = os.path.join("/mnt/polished-lake/home/sfeucht/causalab-internal/nma/outputs/das", model_name, task_folder, run_id)
    model_path = os.path.join(
        das_path,
        "models",
        f"{subspace_layer}__{token_position}",
        modulename,
        "model"
    )
    print(model_path)
    featurizer = Featurizer.load_modules(model_path)
    subspace = featurizer.featurizer.rotate.weight.detach().cuda()  # (model_dim, subspace_dim)
    print(f"  Subspace shape: {subspace.shape}")
    return subspace 

def load_task_info(task_name: str):
    """Lazily load task config to avoid importing all tasks upfront. For rephrased_templates.ipynb"""
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

    return cfg