"""Causal model for the months task.

Example prompt: "Q: What day is three days after Thursday?\nA:"
Expected output: " Sunday"
"""

import random
from cl_patch import CausalModel

TEMPLATE = "Q: What month is {offset} months after {input}?\nA:"

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
OFFSETS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four"
]
PREMODS = list(range(2, 37)) # 2..36

def inp_to_idx(inp): # "January" : 1, "February" : 2, .. 
    return {d: i+1 for i, d in enumerate(MONTHS)}[inp] 

def num_to_idx(offset): # "one" : 1, "two" : 2, ... 
    return {n : i+1 for i, n in enumerate(OFFSETS)}[offset]

def compute_premod(offset: str, month: str):
    return num_to_idx(offset) + inp_to_idx(month)

def compute_output(premod: int):
    return MONTHS[(premod % 12)-1]

def fill_template(offset: str, month: str) -> str:
    """Fill in the template with the offset and month."""
    return TEMPLATE.format(offset=offset, input=month)


# Causal Model Definition
variables = [
    "offset",        # Input: word form of offset (one-seven)
    "input",         # Input: starting month of the week
    "premod",        # Intermediate value: offset + input before mod is applied. 
    "output",        # Computed: resulting month after addition
    "raw_input",   # The full prompt string
    "raw_output",  # The expected output (result month with leading space)
]

values = {
    "offset": OFFSETS,
    "input": MONTHS,
    "premod": PREMODS,
    "output": MONTHS,
    "raw_input": None,
    "raw_output": None,
}

parents = {
    "offset": [],
    "input": [],
    "premod": ["offset", "input"],
    "output": ["premod"],
    "raw_input": ["offset", "input"],
    "raw_output": ["output"],
}

mechanisms = {
    "offset": lambda: random.choice(OFFSETS),
    "input": lambda: random.choice(MONTHS),
    "premod": compute_premod, 
    "output": compute_output,
    "raw_input": fill_template,
    "raw_output": lambda output: " " + output
}

months_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id="months"
)
