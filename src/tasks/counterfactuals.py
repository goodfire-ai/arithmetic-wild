"""Shared counterfactual generators for all tasks.

from tasks.counterfactuals import random_counterfactual
from tasks.weekdays import weekdays_causal_model
cf = random_counterfactual(weekdays_causal_model)
"""

from causalab.causal.causal_model import CausalModel


def random_counterfactual(causal_model: CausalModel):
    """Generate a random counterfactual by sampling two independent inputs."""
    input_sample = causal_model.sample_input()
    counterfactual = causal_model.sample_input()
    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}

def random_counterfactual_batch(causal_model: CausalModel, n: int):
    """Generate n random counterfactual pairs."""
    return [random_counterfactual(causal_model) for _ in range(n)]