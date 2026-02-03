"""Switching network types and base class for network implementations.

NetworkType enumerates available formulations. ISwitchingNetwork is the abstract
base for networks that produce a list of Switch elements.
"""

from abc import ABC, abstractmethod
from enum import StrEnum

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.switch import Switch


class NetworkType(StrEnum):
    """Identifier for each switching network (or naive) formulation."""

    NAIVE = "naive"
    BENES = "benes"
    BITONIC_SORT = "bitonic_sort"
    BUBBLE_SORT = "bubble_sort"
    CLOS_NETWORK_MAX_DEGREE = "clos_network_max_degree"
    CLOS_NETWORK_MIN_EDGE = "clos_network_min_edge"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    ODDEVEN_MERGE_SORT = "oddeven_merge_sort"


class ISwitchingNetwork(ABC):
    """Abstract base for switching networks that produce a list of Switch elements."""

    @classmethod
    @abstractmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[Switch]:
        """Return the raw list of Switch elements for the given left/right nodes."""
        pass

    @classmethod
    def generate_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[Switch]:
        """Build the switching network, simplifying switches when nodes are fixed (ALWAYS_ZERO/ALWAYS_ONE)."""
        network: list[Switch] = cls._generate_original_network(left_nodes, right_nodes, threshold, reverse)

        # Place switch while managing the set of rightmost nodes
        current_nodes: set[str] = {node.name for node in right_nodes}
        name_to_attribute: dict[str, NodeAttribute] = {node.name: node.attribute for node in right_nodes}
        result_network: list[Switch] = []
        for switch in network[::-1]:  # Look from the right
            # Raise an error if there are no nodes to connect
            if not switch.right_nodes.issubset(current_nodes):
                raise ValueError(f"Invalid network: {switch.right_nodes} is not subset of {current_nodes}")
            current_nodes.difference_update(switch.right_nodes)

            # Raise an error if nodes that should be newly generated already exist
            if not switch.left_nodes.isdisjoint(current_nodes):
                raise ValueError(f"Invalid network: {switch.left_nodes} is not disjoint with {current_nodes}")
            current_nodes.update(switch.left_nodes)

            # Calculate the range of possible values for left and right
            num_left_variables: int = len(switch.left_nodes)
            right_sum_min: int = (
                len([node for node in switch.right_nodes if name_to_attribute[node] == NodeAttribute.ALWAYS_ONE])
                + switch.right_constant
                - switch.left_constant
            )
            right_sum_max: int = (
                len([node for node in switch.right_nodes if name_to_attribute[node] != NodeAttribute.ALWAYS_ZERO])
                + switch.right_constant
                - switch.left_constant
            )

            # Raise an error if the range is not achievable
            if not (right_sum_max >= 0 and right_sum_min <= num_left_variables):
                raise ValueError(
                    f"Invalid network: right_sum_max = {right_sum_max} < 0 or right_sum_min = {right_sum_min} > {num_left_variables}"
                )

            # When the right side's lower bound matches the left side's upper bound, all left nodes must be 1
            if right_sum_min == num_left_variables:
                for node in switch.left_nodes:
                    name_to_attribute[node] = NodeAttribute.ALWAYS_ONE
                    result_network.append(
                        Switch(
                            left_nodes=frozenset([node]),
                            right_nodes=frozenset(),
                            left_constant=0,
                            right_constant=1,
                        )
                    )
            # When the right side's upper bound is 0, all left nodes must be 0
            elif right_sum_max == 0:
                for node in switch.left_nodes:
                    name_to_attribute[node] = NodeAttribute.ALWAYS_ZERO
                    result_network.append(
                        Switch(
                            left_nodes=frozenset([node]),
                            right_nodes=frozenset(),
                            left_constant=0,
                            right_constant=0,
                        )
                    )
            # When all right nodes are NOT_CARE and there are no restrictions on the left side's possible value range, all left nodes become NOT_CARE
            elif (
                all(name_to_attribute[node] == NodeAttribute.NOT_CARE for node in switch.right_nodes)
                and right_sum_min <= 0
                and right_sum_max >= num_left_variables
            ):
                for node in switch.left_nodes:
                    name_to_attribute[node] = NodeAttribute.NOT_CARE
            # Otherwise, left nodes become normal nodes
            else:
                for node in switch.left_nodes:
                    name_to_attribute[node] = NodeAttribute.ZERO_OR_ONE
                # Add network with constant nodes omitted
                result_network.append(
                    Switch(
                        left_nodes=frozenset(switch.left_nodes),
                        right_nodes=frozenset([
                            node
                            for node in switch.right_nodes
                            if name_to_attribute[node] != NodeAttribute.ALWAYS_ONE
                            and name_to_attribute[node] != NodeAttribute.ALWAYS_ZERO
                        ]),
                        left_constant=switch.left_constant,
                        right_constant=switch.right_constant
                        + len([
                            node for node in switch.right_nodes if name_to_attribute[node] == NodeAttribute.ALWAYS_ONE
                        ]),
                    )
                )
        if reverse:
            return result_network[::-1]
        else:
            return result_network
