from enum import StrEnum

from ..networks.divide_and_conquer_network import DivideAndConquerNetwork
from .node import NodeAttribute, VariableNode
from .permutation_channel import QUBO, PermutationChannel


class ConstraintType(StrEnum):
    ONE_HOT = "one_hot"
    EQUAL_TO = "equal_to"
    LESS_EQUAL = "less_equal"
    GREATER_EQUAL = "greater_equal"
    CLAMP = "clamp"


def get_initial_nodes(
    variables: list[str],
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> tuple[list[VariableNode], list[VariableNode]]:
    size = len(variables)
    left_nodes: list[VariableNode] = [VariableNode(name=v, attribute=NodeAttribute.ZERO_OR_ONE) for v in variables]
    right_nodes: list[VariableNode] = []
    if constraint_type == ConstraintType.ONE_HOT:
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]
    elif constraint_type == ConstraintType.EQUAL_TO:
        if not (c1 is not None and 0 <= c1 <= size):
            raise ValueError("c1 must be between 0 and size")
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - c1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]
    elif constraint_type == ConstraintType.LESS_EQUAL:
        if not (c1 is not None and 0 < c1 <= size):
            raise ValueError("c1 must be between 0 and size")
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - c1 else NodeAttribute.NOT_CARE,
            )
            for i in range(size)
        ]
    elif constraint_type == ConstraintType.GREATER_EQUAL:
        if not (c1 is not None and 0 <= c1 < size):
            raise ValueError("c1 must be between 0 and size")
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.NOT_CARE if i < size - c1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]
    elif constraint_type == ConstraintType.CLAMP:
        if not (c1 is not None and c2 is not None and 0 <= c1 <= c2 <= size):
            raise ValueError("c1 and c2 must be between 0 and size")
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO
                if i < size - c2
                else NodeAttribute.NOT_CARE
                if i < size - c1
                else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]

    else:
        raise NotImplementedError

    return (left_nodes, right_nodes)


def get_constraint_qubo(
    variables: list[str],
    constraint_type: ConstraintType,
    network_name: str = "divide_and_conquer",
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
) -> QUBO:
    left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
    if network_name == "divide_and_conquer":
        channels = DivideAndConquerNetwork.generate_network(
            left_nodes,
            right_nodes,
            threshold=threshold,
        )
    else:
        raise NotImplementedError
    return PermutationChannel.to_qubo(channels)
