from typing import ClassVar

from ..core.node import NodeAttribute, VariableNode
from ..core.permutation_channel import PermutationChannel
from .clos_network_base import ClosNetworkBase


class AdhocNetworkWithMinimumDegree:
    @classmethod
    def implement_if_small(
        cls, left_nodes: list[str], right_nodes: list[str], max_degree: int
    ) -> list[PermutationChannel]:
        N = max(len(left_nodes), len(right_nodes))
        if N < 2:
            raise ValueError("N must be greater than or equal to 2")
        if max_degree >= N:
            return [
                PermutationChannel(
                    left_nodes=frozenset(left_nodes),
                    right_nodes=frozenset(right_nodes),
                )
            ]
        # elif N <= max_degree * 1.5:
        #     first_vars: list[str] = [
        #         f"{left_nodes[i]}_{right_nodes[i]}"
        #         for i in range(math.floor(max_degree / 2))
        #     ]
        #     second_left_vars: list[str] = [
        #         f"{left_nodes[i]}_{i}"
        #         for i in range(math.floor(max_degree / 2), max_degree)
        #     ]
        #     second_right_vars: list[str] = [
        #         f"{right_nodes[i]}_{i}"
        #         for i in range(math.floor(max_degree / 2), max_degree)
        #     ]
        #     return [
        #         PermutationChannel(
        #             left_nodes=frozenset(left_nodes[:max_degree]),
        #             right_nodes=frozenset(first_vars + second_left_vars),
        #         ),
        #         PermutationChannel(
        #             left_nodes=frozenset(second_left_vars + left_nodes[max_degree:]),
        #             right_nodes=frozenset(second_right_vars + right_nodes[max_degree:]),
        #         ),
        #         PermutationChannel(
        #             left_nodes=frozenset(first_vars + second_right_vars),
        #             right_nodes=frozenset(right_nodes[:max_degree]),
        #         ),
        #     ]
        # TODO: Needs confirmation
        else:
            return []


class ClosNetworkWithMaxDegree(ClosNetworkBase):
    num_elements_dict: ClassVar[dict[int, int]] = {}
    max_degree: ClassVar[int | None] = None

    @classmethod
    def reset_max_degree(cls, new_max: int) -> None:
        if new_max < 2:
            raise ValueError("new_max must be greater than or equal to 2")

        cls.num_elements_dict = {}
        cls.max_degree = new_max

    # Number of comparators when implementing a network of size N based on (n, r)
    @classmethod
    def _calc_num_elements(cls, N: int, n: int, r: int) -> int:
        interior_cost = cls._get_estimated_cost_and_implementation(r) * n
        exterior_cost = r
        return exterior_cost * 2 + interior_cost

    # Return the number of comparators
    @classmethod
    def _get_estimated_cost_and_implementation(cls, N: int) -> int:
        if N not in cls.num_elements_dict:
            if adhoc_network := cls._implement_if_small([f"L{i}" for i in range(N)], [f"R{i}" for i in range(N)]):
                cls.num_elements_dict[N] = len(adhoc_network)
            else:
                n_opt, r_opt = cls._determine_channel_sizes(N, N)
                cls.num_elements_dict[N] = cls._calc_num_elements(N, n_opt, r_opt)
        return cls.num_elements_dict[N]

    # Return optimal (n, r)
    @classmethod
    def _determine_channel_sizes(cls, N_left: int, N_right: int) -> tuple[int, int]:
        if cls.max_degree is None:
            raise RuntimeError("max_degree is None")

        N = max(N_left, N_right)

        nr_list: list[tuple[int, int]] = [(n, (N + n - 1) // n) for n in range(2, cls.max_degree + 1)]
        n_opt, r_opt = min(nr_list, key=lambda x: cls._calc_num_elements(N, x[0], x[1]))
        return n_opt, r_opt

    # Return implementation for small cases
    @classmethod
    def _implement_if_small(cls, left_nodes: list[str], right_nodes: list[str]) -> list[PermutationChannel] | None:
        if cls.max_degree is None:
            raise RuntimeError("max_degree is None")

        return AdhocNetworkWithMinimumDegree.implement_if_small(left_nodes, right_nodes, cls.max_degree)


if __name__ == "__main__":
    num_nodes = 7
    left_nodes: list[VariableNode] = [VariableNode(name=f"L{i}") for i in range(num_nodes)]
    right_nodes: list[VariableNode] = [
        VariableNode(
            name=f"R{i}",
            attribute=NodeAttribute.ZERO_OR_ONE,
        )
        for i in range(num_nodes)
    ]
    new_max = 5
    ClosNetworkWithMaxDegree.reset_max_degree(new_max)
    network = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
    print(network)
    print(ClosNetworkWithMaxDegree.num_elements_dict)
