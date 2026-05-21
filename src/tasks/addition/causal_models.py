"""Causal model for the addition task.

The variables should be thought of as a, b, and output. 
To make this match all of the other tasks, we name things: a=`input`, b=`offset`

Example prompt: "3+7="
Expected output: "10"
"""

import random
from cl_patch import CausalModel

# Constants
NUMBERS = [str(i) for i in range(1, 101)]
RESULTS = [str(i) for i in range(2, 201)]

TEMPLATE = "{input}+{offset}="

def inp_to_idx(a: str):
    return int(a)

def num_to_idx(b: str):
    return int(b)

# Causal Model Mechanisms
def compute_result_sum(a: str, b: str) -> str:
    """Compute the sum of a and b.

    Args:
        a: First number as string (e.g., "3")
        b: Second number as string (e.g., "7")

    Returns:
        The sum as a string (e.g., "10")
    """
    return str(int(a) + int(b))


def fill_template(a: str, b: str) -> str:
    """Fill in the template with a and b."""
    return TEMPLATE.format(input=a, offset=b)


# Causal Model Definition
variables = [
    "input",            # Input: first number
    "offset",           # Input: second number
    "output",       # Computed: sum of a and b
    "raw_input",    # The full prompt string
    "raw_output",   # The expected output (sum as string, no leading space)
]

values = {
    "input": NUMBERS,
    "offset": NUMBERS,
    "output": RESULTS,
    "raw_input": None,
    "raw_output": None,
}

parents = {
    "input": [],
    "offset": [],
    "output": ["input", "offset"],
    "raw_input": ["input", "offset"],
    "raw_output": ["output"],
}

mechanisms = {
    "input": lambda: random.choice(NUMBERS),
    "offset": lambda: random.choice(NUMBERS),
    "output": compute_result_sum,
    "raw_input": fill_template,
    "raw_output": lambda output: output,
}

addition_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id="addition"
)
