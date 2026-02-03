from math import log2

from sparse_qubo.core.base_network import ISwitchingNetwork
from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.switch import Switch


# Generate a network representing bitonic sort
class BitonicSortNetwork(ISwitchingNetwork):
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
        if len(left_names) != len(right_names):
            raise ValueError("left_nodes and right_nodes must have the same length")
        N: int = len(left_names)
        n: int = round(log2(N))
        if 2**n != N:
            raise ValueError("N must be a power of 2")
        if not reverse:
            left_names, right_names = right_names, left_names

        all_nodes: list[list[str]] = [[] for _ in range(N)]
        for i in range(N):
            all_nodes[i].append(left_names[i])
            for j in range(n * (n + 1) // 2 - 1):
                all_nodes[i].append(f"{left_names[i]}_{j}_{right_names[i]}")
            all_nodes[i].append(right_names[i])

        progress: list[int] = [0] * N
        result_switches: list[Switch] = []
        # m is called in the order (0, 1, 2, 3), (0, 1, 2), (0, 1), (0)
        if reverse:
            for m_max in range(n)[::-1]:
                for m in range(m_max + 1):
                    M = 2**m
                    for i in range(N):
                        if (i // M) % 2 == 0:
                            result_switches.append(
                                Switch(
                                    left_nodes=frozenset([
                                        all_nodes[i][progress[i]],
                                        all_nodes[i + M][progress[i + M]],
                                    ]),
                                    right_nodes=frozenset([
                                        all_nodes[i][progress[i] + 1],
                                        all_nodes[i + M][progress[i + M] + 1],
                                    ]),
                                )
                            )
                            progress[i] += 1
                            progress[i + M] += 1
            return result_switches
        else:
            for m_max in range(n)[::-1]:
                for m in range(m_max + 1):
                    M = 2**m
                    for i in range(N):
                        if (i // M) % 2 == 0:
                            result_switches.append(
                                Switch(
                                    left_nodes=frozenset([
                                        all_nodes[i][progress[i] + 1],
                                        all_nodes[i + M][progress[i + M] + 1],
                                    ]),
                                    right_nodes=frozenset([
                                        all_nodes[i][progress[i]],
                                        all_nodes[i + M][progress[i + M]],
                                    ]),
                                )
                            )
                            progress[i] += 1
                            progress[i + M] += 1
            return result_switches[::-1]


if __name__ == "__main__":
    import pprint

    left_nodes: list[VariableNode] = [VariableNode(name=f"L{i}") for i in range(8)]
    right_nodes: list[VariableNode] = [
        VariableNode(
            name=f"R{i}",
            attribute=NodeAttribute.ALWAYS_ONE if i < 1 else NodeAttribute.ALWAYS_ZERO,
        )
        for i in range(8)
    ]
    pprint.pprint(BitonicSortNetwork.generate_network(left_nodes, right_nodes))
