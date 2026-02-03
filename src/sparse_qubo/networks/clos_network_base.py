from abc import ABC, abstractmethod

from sparse_qubo.core.network import ISwitchingNetwork
from sparse_qubo.core.node import VariableNode
from sparse_qubo.core.switch import Switch


# Generate a Clos network
class ClosNetworkBase(ISwitchingNetwork, ABC):
    # Return implementation for small cases
    @classmethod
    @abstractmethod
    def _implement_if_small(cls, left_nodes: list[str], right_nodes: list[str]) -> list[Switch] | None:
        pass

    # Determine the values of n and r
    @classmethod
    @abstractmethod
    def _determine_switch_sizes(cls, N_left: int, N_right: int) -> tuple[int, int]:
        pass

    @classmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[Switch]:
        left_names = [node.name for node in left_nodes]
        right_names = [node.name for node in right_nodes]

        if result := cls._implement_if_small(left_names, right_names):
            return result

        left_size: int = len(left_names)
        right_size: int = len(right_names)
        exterior_switch_size, interior_switch_size = cls._determine_switch_sizes(left_size, right_size)
        intermediate_node_size: int = exterior_switch_size * interior_switch_size
        if max(left_size, right_size) > intermediate_node_size:
            raise ValueError("switch size is too small")

        # Generate switches for the ingress stage
        ingress_stage_switches: list[Switch] = []
        ingress_stage_nodes: list[str] = []
        interior_index_start: int
        interior_index_end: int
        interior_node_names: list[str]
        for r in range(interior_switch_size):
            left_index_start: int = r * left_size // interior_switch_size
            left_index_end: int = (r + 1) * left_size // interior_switch_size
            interior_index_start = exterior_switch_size * r
            interior_index_end = exterior_switch_size * (r + 1)

            interior_node_names = [
                f"{left_names[min(i, left_index_end - 1)]}_{i}" for i in range(interior_index_start, interior_index_end)
            ]
            ingress_stage_switches.append(
                Switch(
                    left_nodes=frozenset(left_names[left_index_start:left_index_end]),
                    right_nodes=frozenset(interior_node_names),
                )
            )
            ingress_stage_nodes.extend(interior_node_names)

        # Generate switches for the egress stage
        egress_stage_switches: list[Switch] = []
        egress_stage_nodes: list[str] = []
        for r in range(interior_switch_size):
            right_index_start: int = r * right_size // interior_switch_size
            right_index_end: int = (r + 1) * right_size // interior_switch_size
            interior_index_start = exterior_switch_size * r
            interior_index_end = exterior_switch_size * (r + 1)

            interior_node_names = [
                f"{right_names[min(i, right_index_end - 1)]}_{i}"
                for i in range(interior_index_start, interior_index_end)
            ]
            egress_stage_switches.append(
                Switch(
                    left_nodes=frozenset(interior_node_names),
                    right_nodes=frozenset(right_names[right_index_start:right_index_end]),
                )
            )
            egress_stage_nodes.extend(interior_node_names)

        # # Generate switches for the middle stage
        # middle_stage_switches: list[Switch] = []
        # for i_start in range(exterior_switch_size):
        #     middle_stage_switches.extend(
        #         cls._generate_original_network(
        #             ingress_stage_nodes[i_start:intermediate_node_size:exterior_switch_size],
        #             egress_stage_nodes[i_start:intermediate_node_size:exterior_switch_size],
        #         )
        #     )

        # Generate switches for the middle stage
        middle_stage_switches: list[Switch] = []
        for i_start in range(exterior_switch_size):
            # 1. First, get string slices (sub-lists)
            sub_left_names = ingress_stage_nodes[i_start:intermediate_node_size:exterior_switch_size]
            sub_right_names = egress_stage_nodes[i_start:intermediate_node_size:exterior_switch_size]

            # 2. Convert to VariableNode and make recursive call
            middle_stage_switches.extend(
                cls._generate_original_network(
                    left_nodes=[VariableNode(name=n) for n in sub_left_names],
                    right_nodes=[VariableNode(name=n) for n in sub_right_names],
                    threshold=threshold,
                    reverse=reverse,
                )
            )

        # Connect the three and return
        return ingress_stage_switches + middle_stage_switches + egress_stage_switches
