from typing import ClassVar

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.permutation_channel import PermutationChannel
from sparse_qubo.networks.clos_network_base import ClosNetworkBase


class ClosNetworkMinimumEdge(ClosNetworkBase):
    is_small_dict: ClassVar[dict[int, bool]] = dict.fromkeys(range(3), True)
    num_logical_edges_dict: ClassVar[dict[int, int]] = {0: 0, 1: 0, 2: 6}

    # Number of logical edges when implementing a network of size N based on (n, r)
    @classmethod
    def _calc_num_logical_edges(cls, N: int, n: int, r: int) -> int:
        interior_cost = cls._get_estimated_cost_and_implementation(r)[0] * n
        exterior_cost = 0
        for r_idx in range(r):
            input_start_idx = N * r_idx // r
            input_end_idx = N * (r_idx + 1) // r
            total_nodes = input_end_idx - input_start_idx + n
            exterior_cost += total_nodes * (total_nodes - 1) // 2
        return exterior_cost * 2 + interior_cost

    # Return the number of logical edges and implementation method (whether clique or not)
    @classmethod
    def _get_estimated_cost_and_implementation(cls, N: int) -> tuple[int, bool]:
        if N not in cls.num_logical_edges_dict:
            n_opt, r_opt = cls._determine_channel_sizes(N, N)
            cost_division = cls._calc_num_logical_edges(N, n_opt, r_opt)
            cost_clique = N * (N * 2 - 1)
            cls.is_small_dict[N] = cost_clique <= cost_division
            cls.num_logical_edges_dict[N] = min(cost_clique, cost_division)
        return cls.num_logical_edges_dict[N], cls.is_small_dict[N]

    # Return optimal (n, r)
    @classmethod
    def _determine_channel_sizes(cls, N_left: int, N_right: int) -> tuple[int, int]:
        N = max(N_left, N_right)

        # r = (N + n - 1) // n is the minimum r that satisfies n*r >= N
        nr_list: list[tuple[int, int]] = [(n, (N + n - 1) // n) for n in range(2, N)]
        n_opt, r_opt = min(nr_list, key=lambda x: cls._calc_num_logical_edges(N, x[0], x[1]))
        return n_opt, r_opt

    # Return implementation for small cases
    @classmethod
    def _implement_if_small(cls, left_nodes: list[str], right_nodes: list[str]) -> list[PermutationChannel] | None:
        N = max(len(left_nodes), len(right_nodes))
        is_small = cls._get_estimated_cost_and_implementation(N)[1]
        if is_small:
            return [
                PermutationChannel(
                    left_nodes=frozenset(left_nodes),
                    right_nodes=frozenset(right_nodes),
                )
            ]
        else:
            return None


if __name__ == "__main__":
    num_nodes = 8
    left_nodes: list[VariableNode] = [VariableNode(name=f"L{i}") for i in range(num_nodes)]
    right_nodes: list[VariableNode] = [
        VariableNode(
            name=f"R{i}",
            attribute=NodeAttribute.ZERO_OR_ONE,
        )
        for i in range(num_nodes)
    ]
    network = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
    print(ClosNetworkMinimumEdge.is_small_dict)
    print(ClosNetworkMinimumEdge.num_logical_edges_dict)
