from .weekdays import (
    weekdays_causal_model,
    weekdays_token_positions,
    inp_to_idx as weekdays_inp_to_idx,
    num_to_idx as weekdays_num_to_idx,
)
from .months import (
    months_causal_model,
    months_token_positions,
    inp_to_idx as months_inp_to_idx,
    num_to_idx as months_num_to_idx,
)
from .hours import (
    hours_causal_model,
    hours_token_positions,
    inp_to_idx as hrs_inp_to_idx,
    num_to_idx as hrs_num_to_idx,
)
from .addition import (
    addition_causal_model,
    addition_token_positions,
    inp_to_idx as add_inp_to_idx,
    num_to_idx as add_num_to_idx,
)
from .counterfactuals import random_counterfactual

TASKS = {
    "weekdays": {
        "causal_model": weekdays_causal_model,
        "create_token_positions": weekdays_token_positions,
        "single_token_important": ["offset", "input"],
        "inp_to_idx": weekdays_inp_to_idx,
        "num_to_idx": weekdays_num_to_idx
    },
    "months" : {
        "causal_model": months_causal_model,
        "create_token_positions": months_token_positions,
        "single_token_important": ["offset", "input"],
        "inp_to_idx": months_inp_to_idx,
        "num_to_idx": months_num_to_idx
    },
    "hours" : {
        "causal_model": hours_causal_model,
        "create_token_positions": hours_token_positions,
        "single_token_important": ["offset", "input"],
        "inp_to_idx": hrs_inp_to_idx,
        "num_to_idx": hrs_num_to_idx
    },
    "addition" : {
        "causal_model": addition_causal_model,
        "create_token_positions": addition_token_positions,
        "single_token_important": ["a", "input"],
        "inp_to_idx": add_inp_to_idx,
        "num_to_idx": add_num_to_idx
    }
}

__all__ = [
    "TASKS",
]