"""Causal model for the weekdays task.

Example prompt: "Q: What day is three days after Thursday?\nA:"
Expected output: " Sunday"
"""

import random
from cl_patch import CausalModel

TEMPLATE = "Q: What day is {offset} days after {input}?\nA:"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OFFSETS = [
    "one", "two", "three", "four", "five", "six", "seven", 
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"
]
PREMODS = list(range(2, 22)) # 2..21

def inp_to_idx(inp): # "Monday" : 1, "Tuesday" : 2, .. 
    return {d: i+1 for i, d in enumerate(DAYS)}[inp] 

def num_to_idx(offset): # "one" : 1, "two" : 2, ... 
    return {n : i+1 for i, n in enumerate(OFFSETS)}[offset]

def compute_premod(offset: str, day: str):
    return num_to_idx(offset) + inp_to_idx(day)

def compute_output(premod: int):
    return DAYS[(premod % 7) - 1]

def fill_template(offset: str, day: str) -> str:
    """Fill in the template with the offset and day."""
    return TEMPLATE.format(offset=offset, input=day)


# Causal Model Definition
variables = [
    "offset",        # Input: word form of offset (one-seven)
    "input",         # Input: starting day of the week
    "premod",        # Intermediate value: offset + input before mod is applied. 
    "output",        # Computed: resulting day after addition
    "raw_input",   # The full prompt string
    "raw_output",  # The expected output (result day with leading space)
]

values = {
    "offset": OFFSETS,
    "input": DAYS,
    "premod": PREMODS,
    "output": DAYS,
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
    "input": lambda: random.choice(DAYS),
    "premod": compute_premod, 
    "output": compute_output,
    "raw_input": fill_template,
    "raw_output": lambda output: " " + output
}

weekdays_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id="weekdays"
)
