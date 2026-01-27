from enum import StrEnum

from sparse_qubo.core.base_network import NetworkType
from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.permutation_channel import QUBO, PermutationChannel
from sparse_qubo.networks.benes_network import BenesNetwork
from sparse_qubo.networks.bitonic_sort_network import BitonicSortNetwork
from sparse_qubo.networks.bubble_sort_network import BubbleSortNetwork
from sparse_qubo.networks.clos_network_max_degree import ClosNetworkWithMaxDegree
from sparse_qubo.networks.clos_network_minimum_edge import ClosNetworkMinimumEdge
from sparse_qubo.networks.divide_and_conquer_network import DivideAndConquerNetwork
from sparse_qubo.networks.oddeven_merge_sort_network import OddEvenMergeSortNetwork


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
    exponentiation: bool = False,  # 追加
) -> tuple[list[VariableNode], list[VariableNode]]:
    original_size = len(variables)
    target_size = original_size
    if exponentiation and original_size > 0:
        target_size = 1 << (original_size - 1).bit_length()
    pad_len = target_size - original_size

    left_attrs = [NodeAttribute.ALWAYS_ZERO] * pad_len + [NodeAttribute.ZERO_OR_ONE] * original_size

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

    left_nodes = [VariableNode(name=f"L{i}", attribute=attr) for i, attr in enumerate(left_attrs)]
    right_nodes = [VariableNode(name=f"R{i}", attribute=attr) for i, attr in enumerate(right_attrs)]

    return left_nodes, right_nodes


def get_constraint_qubo(
    variables: list[str],
    constraint_type: ConstraintType,
    network_type: NetworkType = NetworkType.DIVIDE_AND_CONQUER,
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
    reverse: bool = False,
) -> QUBO:
    match network_type:
        case NetworkType.BENES:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2, exponentiation=True)
            channels = BenesNetwork.generate_network(left_nodes, right_nodes, threshold=threshold, reverse=reverse)
        case NetworkType.BITONIC_SORT:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2, exponentiation=True)
            channels = BitonicSortNetwork.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.BUBBLE_SORT:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            channels = BubbleSortNetwork.generate_network(left_nodes, right_nodes, threshold=threshold, reverse=reverse)
        case NetworkType.CLOS_NETWORK_MAX_DEGREE:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            channels = ClosNetworkWithMaxDegree.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.CLOS_NETWORK_MIN_EDGE:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            channels = ClosNetworkMinimumEdge.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.DIVIDE_AND_CONQUER:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2)
            channels = DivideAndConquerNetwork.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case NetworkType.ODDEVEN_MERGE_SORT:
            left_nodes, right_nodes = get_initial_nodes(variables, constraint_type, c1, c2, exponentiation=True)
            channels = OddEvenMergeSortNetwork.generate_network(
                left_nodes, right_nodes, threshold=threshold, reverse=reverse
            )
        case _:
            raise NotImplementedError
    return PermutationChannel.to_qubo(channels)
