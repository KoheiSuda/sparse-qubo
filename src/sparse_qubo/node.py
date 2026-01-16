from enum import Enum

from pydantic import BaseModel, Field


# Attributes of binary variables
class NodeAttribute(Enum):
    ZERO_OR_ONE = "ZERO_OR_ONE"  # General case
    ALWAYS_ZERO = "ALWAYS_ZERO"  # Can only take 0
    ALWAYS_ONE = "ALWAYS_ONE"  # Can only take 1
    NOT_CARE = "NOT_CARE"  # Not related to constraints


# Node representing a binary variable
class VariableNode(BaseModel):
    name: str = Field(description="Name of the binary variable", frozen=True)
    attribute: NodeAttribute = Field(default=NodeAttribute.ZERO_OR_ONE, description="Attribute of the binary variable")
