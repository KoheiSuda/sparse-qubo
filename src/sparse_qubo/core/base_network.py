from abc import ABC, abstractmethod
from enum import StrEnum

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.permutation_channel import PermutationChannel


class NetworkType(StrEnum):
    NAIVE = "naive"
    BENES = "benes"
    BITONIC_SORT = "bitonic_sort"
    BUBBLE_SORT = "bubble_sort"
    CLOS_NETWORK_MAX_DEGREE = "clos_network_max_degree"
    CLOS_NETWORK_MIN_EDGE = "clos_network_min_edge"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    ODDEVEN_MERGE_SORT = "oddeven_merge_sort"


# Abstract base class for switching networks
class ISwitchingNetwork(ABC):
    @classmethod
    @abstractmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[PermutationChannel]:
        pass

    # Generate a network with variables that can be fixed to scalars removed
    @classmethod
    def generate_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[PermutationChannel]:
        network: list[PermutationChannel] = cls._generate_original_network(left_nodes, right_nodes, threshold, reverse)

        # Place channels while managing the set of rightmost nodes
        current_nodes: set[str] = {node.name for node in right_nodes}
        name_to_attribute: dict[str, NodeAttribute] = {node.name: node.attribute for node in right_nodes}
        result_network: list[PermutationChannel] = []
        for channel in network[::-1]:  # Look from the right
            # Raise an error if there are no nodes to connect
            if not channel.right_nodes.issubset(current_nodes):
                raise ValueError(f"Invalid network: {channel.right_nodes} is not subset of {current_nodes}")
            current_nodes.difference_update(channel.right_nodes)

            # Raise an error if nodes that should be newly generated already exist
            if not channel.left_nodes.isdisjoint(current_nodes):
                raise ValueError(f"Invalid network: {channel.left_nodes} is not disjoint with {current_nodes}")
            current_nodes.update(channel.left_nodes)

            # Calculate the range of possible values for left and right
            num_left_variables: int = len(channel.left_nodes)
            right_sum_min: int = (
                len([node for node in channel.right_nodes if name_to_attribute[node] == NodeAttribute.ALWAYS_ONE])
                + channel.right_constant
                - channel.left_constant
            )
            right_sum_max: int = (
                len([node for node in channel.right_nodes if name_to_attribute[node] != NodeAttribute.ALWAYS_ZERO])
                + channel.right_constant
                - channel.left_constant
            )

            # Raise an error if the range is not achievable
            if not (right_sum_max >= 0 and right_sum_min <= num_left_variables):
                raise ValueError(
                    f"Invalid network: right_sum_max = {right_sum_max} < 0 or right_sum_min = {right_sum_min} > {num_left_variables}"
                )

            # When the right side's lower bound matches the left side's upper bound, all left nodes must be 1
            if right_sum_min == num_left_variables:
                for node in channel.left_nodes:
                    name_to_attribute[node] = NodeAttribute.ALWAYS_ONE
                    result_network.append(
                        PermutationChannel(
                            left_nodes=frozenset([node]),
                            right_nodes=frozenset(),
                            left_constant=0,
                            right_constant=1,
                        )
                    )
            # When the right side's upper bound is 0, all left nodes must be 0
            elif right_sum_max == 0:
                for node in channel.left_nodes:
                    name_to_attribute[node] = NodeAttribute.ALWAYS_ZERO
                    result_network.append(
                        PermutationChannel(
                            left_nodes=frozenset([node]),
                            right_nodes=frozenset(),
                            left_constant=0,
                            right_constant=0,
                        )
                    )
            # When all right nodes are NOT_CARE and there are no restrictions on the left side's possible value range, all left nodes become NOT_CARE
            elif (
                all(name_to_attribute[node] == NodeAttribute.NOT_CARE for node in channel.right_nodes)
                and right_sum_min <= 0
                and right_sum_max >= num_left_variables
            ):
                for node in channel.left_nodes:
                    name_to_attribute[node] = NodeAttribute.NOT_CARE
            # Otherwise, left nodes become normal nodes
            else:
                for node in channel.left_nodes:
                    name_to_attribute[node] = NodeAttribute.ZERO_OR_ONE
                # Add network with constant nodes omitted
                result_network.append(
                    PermutationChannel(
                        left_nodes=frozenset(channel.left_nodes),
                        right_nodes=frozenset([
                            node
                            for node in channel.right_nodes
                            if name_to_attribute[node] != NodeAttribute.ALWAYS_ONE
                            and name_to_attribute[node] != NodeAttribute.ALWAYS_ZERO
                        ]),
                        left_constant=channel.left_constant,
                        right_constant=channel.right_constant
                        + len([
                            node for node in channel.right_nodes if name_to_attribute[node] == NodeAttribute.ALWAYS_ONE
                        ]),
                    )
                )
        if reverse:
            return result_network[::-1]
        else:
            return result_network
