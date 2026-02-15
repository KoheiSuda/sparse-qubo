"""Constraint types and QUBO construction for equality/inequality constraints.

This module provides ConstraintType, get_initial_nodes and get_constraint_switches to build Switches from
switching networks for use with D-Wave or Amplify.
"""

from enum import StrEnum

from sparse_qubo.core.network import NetworkType
from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.switch import Switch
from sparse_qubo.networks.benes_network import BenesNetwork
from sparse_qubo.networks.bitonic_sort_network import BitonicSortNetwork
from sparse_qubo.networks.bubble_sort_network import BubbleSortNetwork
from sparse_qubo.networks.clos_network_max_degree import ClosNetworkWithMaxDegree
from sparse_qubo.networks.clos_network_minimum_edge import ClosNetworkMinimumEdge
from sparse_qubo.networks.divide_and_conquer_network import DivideAndConquerNetwork
from sparse_qubo.networks.oddeven_merge_sort_network import OddEvenMergeSortNetwork

# Internal counter for auto-assigning unique prefixes to auxiliary variables.
# Incremented on each get_constraint_switches(..., var_prefix=None) call.
_constraint_prefix_counter = 0


def reset_constraint_prefix_counter() -> None:
    """Reset the internal constraint prefix counter.

    The prefix counter is **not** reset automatically. It is reset only when:

    1. **Process start**: Restarting Python (e.g. re-running a script or
       restarting a Jupyter kernel) reloads the module and sets the counter to 0.
    2. **Explicit call**: Calling this function sets the counter back to 0.

    Call this when you want the next constraint to use prefix ``C0`` again
    (e.g. when starting to build a new model in the same process). Mainly useful
    for testing or reproducible variable names.
    """
    global _constraint_prefix_counter
    _constraint_prefix_counter = 0


class ConstraintType(StrEnum):
    """Type of linear constraint on binary variables (sum of variables)."""

    ONE_HOT = "one_hot"  # Exactly one variable is 1
    EQUAL_TO = "equal_to"  # Sum equals c1
    LESS_EQUAL = "less_equal"  # Sum <= c1
    GREATER_EQUAL = "greater_equal"  # Sum >= c1
    CLAMP = "clamp"  # c1 <= sum <= c2


def get_initial_nodes(
    variables: list[str],
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
    exponentiation: bool = False,
) -> tuple[list[VariableNode], list[VariableNode]]:
    """Build left and right VariableNode lists for a switching network from variable names and constraint type.

    Used by get_constraint_switches and by network implementations. If exponentiation is True,
    the right side is padded to a power-of-2 size (for Benes, Bitonic, OddEvenMergeSort).
    """
    original_size = len(variables)
    target_size = original_size
    if exponentiation and original_size > 0:
        target_size = 1 << (original_size - 1).bit_length()
    pad_len = target_size - original_size

    left_pad_attrs = [NodeAttribute.ALWAYS_ZERO] * pad_len
    left_var_attrs = [NodeAttribute.ZERO_OR_ONE] * original_size

    def get_original_right_attr(i: int) -> NodeAttribute:
        match constraint_type:
            case ConstraintType.ONE_HOT:
                return NodeAttribute.ALWAYS_ZERO if i < original_size - 1 else NodeAttribute.ALWAYS_ONE
            case ConstraintType.EQUAL_TO:
                if c1 is None or not (0 <= c1 <= original_size):
                    raise ValueError(f"c1 must be between 0 and {original_size}")
                return NodeAttribute.ALWAYS_ZERO if i < original_size - c1 else NodeAttribute.ALWAYS_ONE
            case ConstraintType.LESS_EQUAL:
                if c1 is None or not (0 < c1 <= original_size):
                    raise ValueError(f"c1 must be between 0 and {original_size}")
                return NodeAttribute.ALWAYS_ZERO if i < original_size - c1 else NodeAttribute.NOT_CARE
            case ConstraintType.GREATER_EQUAL:
                if c1 is None or not (0 <= c1 < original_size):
                    raise ValueError(f"c1 must be between 0 and {original_size}")
                return NodeAttribute.NOT_CARE if i < original_size - c1 else NodeAttribute.ALWAYS_ONE
            case ConstraintType.CLAMP:
                if c1 is None or c2 is None or not (0 <= c1 <= c2 <= original_size):
                    raise ValueError(f"c1 and c2 must be valid range (0 <= c1 <= c2 <= {original_size})")
                if i < original_size - c2:
                    return NodeAttribute.ALWAYS_ZERO
                elif i < original_size - c1:
                    return NodeAttribute.NOT_CARE
                else:
                    return NodeAttribute.ALWAYS_ONE
            case _:
                raise NotImplementedError(f"Constraint type {constraint_type} is not supported")

    original_right_attrs = [get_original_right_attr(i) for i in range(original_size)]
    right_attrs = [NodeAttribute.ALWAYS_ZERO] * pad_len + original_right_attrs

    left_nodes = [VariableNode(name=f"L{i}", attribute=attr) for i, attr in enumerate(left_pad_attrs)] + [
        VariableNode(name=str(v), attribute=attr) for v, attr in zip(variables, left_var_attrs, strict=True)
    ]
    right_nodes = [VariableNode(name=f"R{i}", attribute=attr) for i, attr in enumerate(right_attrs)]

    return left_nodes, right_nodes


def _prefix_auxiliary_variables(switches: list[Switch], original_variables: set[str], prefix: str) -> list[Switch]:
    """Rename all auxiliary variables (not in original_variables) by adding prefix."""

    def renamed(v: str) -> str:
        return f"{prefix}_{v}" if v not in original_variables else v

    return [
        Switch(
            left_nodes=frozenset([renamed(v) for v in switch.left_nodes]),
            right_nodes=frozenset([renamed(v) for v in switch.right_nodes]),
            left_constant=switch.left_constant,
            right_constant=switch.right_constant,
        )
        for switch in switches
    ]


def get_constraint_switches(
    variables: list[str],
    constraint_type: ConstraintType,
    network_type: NetworkType = NetworkType.DIVIDE_AND_CONQUER,
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
    reverse: bool = False,
    var_prefix: str | None = None,
) -> list[Switch]:
    """Build a list of Switches for the given constraint.

    **Auxiliary variable prefixes**

    When ``var_prefix`` is ``None`` (the default), a unique prefix is assigned
    internally (``C0``, ``C1``, ...) so that merging multiple constraint QUBOs
    into one BQM avoids name collisions. The counter increments on each call
    and is **not** reset automatically; it resets only on process start or when
    calling :func:`reset_constraint_prefix_counter`. See the Usage section
    (Constraint prefix counter) in the documentation for details.
    """
    global _constraint_prefix_counter
    if var_prefix is None:
        var_prefix = f"C{_constraint_prefix_counter!s}"
        _constraint_prefix_counter += 1

    match network_type:
        case NetworkType.BENES:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2, exponentiation=True)
            switches = BenesNetwork.generate_network(left_nodes, right_nodes, threshold=threshold, reverse=reverse)
        case NetworkType.BITONIC_SORT:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2, exponentiation=True)
            switches = BitonicSortNetwork.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.BUBBLE_SORT:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes, threshold=threshold, reverse=reverse)
        case NetworkType.CLOS_NETWORK_MAX_DEGREE:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            switches = ClosNetworkWithMaxDegree.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.CLOS_NETWORK_MIN_EDGE:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            switches = ClosNetworkMinimumEdge.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.DIVIDE_AND_CONQUER:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            switches = DivideAndConquerNetwork.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.ODDEVEN_MERGE_SORT:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2, exponentiation=True)
            switches = OddEvenMergeSortNetwork.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case _:
            raise NotImplementedError
    return _prefix_auxiliary_variables(switches, set(variables), var_prefix)
