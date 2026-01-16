from enum import Enum

import dimod
import dimod.variables

from ..networks.divide_and_conquer_network import DivideAndConquerNetwork
from ..node import NodeAttribute, VariableNode
from ..permutation_channel import PermutationChannel


class ConstraintType(Enum):
    ONE_HOT = "one_hot"
    EQUAL_TO = "equal_to"
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    CLAMP = "clamp"


def naive_constraint(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> dimod.BinaryQuadraticModel:
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    size = len(variables)
    if constraint_type == ConstraintType.ONE_HOT:
        bqm.add_linear_equality_constraint(
            [(str(v), 1) for v in variables],
            1,
            -1,
        )
        return bqm
    if constraint_type == ConstraintType.EQUAL_TO:
        if not (c1 is not None and 0 <= c1 <= size):
            raise ValueError("c1 must be between 0 and size")
        bqm.add_linear_equality_constraint(
            [(str(v), 1) for v in variables],
            1,
            -c1,
        )
        return bqm
    else:
        raise NotImplementedError


def get_initial_nodes(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> tuple[list[VariableNode], list[VariableNode]]:
    size = len(variables)
    left_nodes: list[VariableNode] = [VariableNode(name=str(v), attribute=NodeAttribute.ZERO_OR_ONE) for v in variables]
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
    elif constraint_type == ConstraintType.LESS_THAN:
        if not (c1 is not None and 0 < c1 <= size):
            raise ValueError("c1 must be between 0 and size")
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - c1 else NodeAttribute.NOT_CARE,
            )
            for i in range(size)
        ]
    elif constraint_type == ConstraintType.GREATER_THAN:
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


def constraint(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    network_name: str = "divide_and_conquer",
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
) -> dimod.BinaryQuadraticModel:
    if network_name == "divide_and_conquer":
        left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
        channels = DivideAndConquerNetwork.generate_network(
            left_nodes,
            right_nodes,
            threshold=threshold,
        )
        qubo = PermutationChannel.to_qubo(channels)
        bqm = dimod.BinaryQuadraticModel(
            qubo.linear,
            qubo.quadratic,
            qubo.constant,
            dimod.BINARY,
        )
        return bqm
    elif network_name == "naive":
        bqm = naive_constraint(variables, constraint_type, c1, c2)
        return bqm
    else:
        raise NotImplementedError
