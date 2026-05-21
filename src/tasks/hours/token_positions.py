"""
This module provides functions to locate specific tokens in hours prompts:
- last_token: The last token in the sequence
- offset: The offset word token (one, two, three, etc.)
- hour: The hour token (00-23)
"""

from causalab.neural.token_position_builder import build_token_position_factories, TokenPosition
from causalab.neural.pipeline import LMPipeline
from .causal_models import TEMPLATE

from typing import Any

# Type alias for token position specs
TokenPositionSpec = dict[str, Any]

def hours_token_positions(pipeline: LMPipeline) -> dict[str, TokenPosition]:
    """
    Create all token positions for the hours task.

    Args:
        pipeline: The tokenizer pipeline

    Returns:
        dict: Dictionary mapping token position names to TokenPosition objects
    """
    # Define token position specifications using the declarative system
    token_position_specs: dict[str, TokenPositionSpec] = {
        # The hour (should be a single token like "17" or "09")
        "input": {
            "type": "index",
            "position": -1,
            "scope": {"variable": "input"}
        },

        # The offset word (last token in case MTW, but unlikely)
        "offset": {
            "type": "index",
            "position": -1,
            "scope": {"variable": "offset"}
        },

        # Last token in the sequence
        "last_token": {"type": "index", "position": -1}
    }

    # Build token position factories
    factories = build_token_position_factories(token_position_specs, TEMPLATE)

    # Call each factory with the pipeline to create actual TokenPosition objects
    token_positions = {}
    for name, factory in factories.items():
        token_positions[name] = factory(pipeline)

    return token_positions
