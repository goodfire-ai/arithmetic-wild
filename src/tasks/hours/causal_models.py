"""Causal model for the hours task.

Example prompt: "Q: It is now 17:00. What time will it be in three hours?\nA:"
Expected output: " 20:00"
"""

import random
from cl_patch import CausalModel

# Hours as zero-padded strings for consistent tokenization
TEMPLATE = "Q: In 24-hour time, it is now {input}:00. What time will it be in {offset} hours?\nA: In 24-hour time, it will be "
HOURS = [f"{h:02d}" for h in range(24)]  # "00", "01", ..., "23"
OFFSETS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four",
    "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty", "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five", "thirty-six",
    "thirty-seven", "thirty-eight", "thirty-nine", "forty", "forty-one", "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight"
]
PREMODS = list(range(2, 72)) # 2..(23+48)

def inp_to_idx(inp): # 00 : 0, 01 : 1, 02: 2, ...
    return int(inp)

def num_to_idx(offset): # "one" : 1, "two" : 2, ... 
    return {n : i+1 for i, n in enumerate(OFFSETS)}[offset]

def compute_premod(offset: str, hour: str):
    return num_to_idx(offset) + inp_to_idx(hour)

def compute_output(premod: int) -> str:
    """Compute the resulting hour after adding N hours to the starting hour.

    Args:
        offset: Word form of the offset (e.g., "three")
        hour: Starting hour as zero-padded string (e.g., "17")

    Returns:
        The resulting hour as zero-padded string (e.g., "20")
    """
    result_hour = (premod) % 24
    return f"{result_hour:02d}"


def fill_template(offset: str, hour: str) -> str:
    """Fill in the template with the offset and hour."""
    return TEMPLATE.format(offset=offset, input=hour)


# Causal Model Definition
variables = [
    "offset",       # Input: word form of offset (one-twelve)
    "input",         # Input: starting hour (00-23)
    "premod",       # intermediate value 
    "output",  # Computed: resulting hour after addition
    "raw_input",    # The full prompt string
    "raw_output",   # The expected output (result time with leading space)
]

values = {
    "offset": OFFSETS,
    "input": HOURS,
    "premod": PREMODS,
    "output": HOURS,
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
    "input": lambda: random.choice(HOURS),
    "premod": compute_premod,
    "output": compute_output,
    "raw_input": fill_template,
    "raw_output": lambda result_hour: result_hour + ":00", # no preceding space because template has one 
}

hours_causal_model = CausalModel(
    variables,
    values,
    parents,
    mechanisms,
    id="hours"
)
