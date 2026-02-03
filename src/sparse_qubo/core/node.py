"""Nodes and attributes for switching network construction.

VariableNode represents a binary variable with an optional NodeAttribute
(ALWAYS_ZERO, ALWAYS_ONE, NOT_CARE, or ZERO_OR_ONE).
"""

from enum import Enum

from pydantic import BaseModel, Field


class NodeAttribute(Enum):
    """Attribute of a binary variable in the switching network."""

    ZERO_OR_ONE = "ZERO_OR_ONE"  # General case
    ALWAYS_ZERO = "ALWAYS_ZERO"  # Fixed to 0
    ALWAYS_ONE = "ALWAYS_ONE"  # Fixed to 1
    NOT_CARE = "NOT_CARE"  # Unconstrained


class VariableNode(BaseModel):
    """A binary variable with a name and an attribute."""

    name: str = Field(description="Name of the binary variable", frozen=True)
    attribute: NodeAttribute = Field(default=NodeAttribute.ZERO_OR_ONE, description="Attribute of the binary variable")
