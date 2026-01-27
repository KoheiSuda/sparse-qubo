from math import log2

from sparse_qubo.core.base_network import ISwitchingNetwork
from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.permutation_channel import PermutationChannel


# TODO: The way reverse is handled is not clean
# Generate a network representing Batcher's odd-even merge sort
class OddEvenMergeSortNetwork(ISwitchingNetwork):
    @classmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = True,
    ) -> list[PermutationChannel]:
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

        progress: list[int] = [0] * N
        temp_result_channels: list[PermutationChannel] = []
        # m is called in the order (0, 1, 2, 3), (0, 1, 2), (0, 1), (0)
        for m_max in range(1, n + 1)[::-1]:
            M_max: int = 2**m_max
            for i_base in range(0, N, M_max):
                for m in range(m_max):
                    M: int = 2**m
                    i_start: int = i_base if m < m_max - 1 else i_base - M
                    i_end: int = i_base + M_max - M
                    for i in range(i_start, i_end):
                        if (i - i_start) // M % 2 == 1:
                            temp_result_channels.append(
                                PermutationChannel(
                                    left_nodes=frozenset([
                                        f"{i}_{progress[i]}",
                                        f"{i + M}_{progress[i + M]}",
                                    ]),
                                    right_nodes=frozenset([
                                        f"{i}_{progress[i] + 1}",
                                        f"{i + M}_{progress[i + M] + 1}",
                                    ]),
                                )
                            )
                            progress[i] += 1
                            progress[i + M] += 1
        temp_name_to_result_name: dict[str, str] = {}
        for i in range(N):
            for j in range(progress[i] + 1):
                if j == 0:
                    temp_name_to_result_name[f"{i}_{j}"] = left_names[i]
                elif j == progress[i]:
                    temp_name_to_result_name[f"{i}_{j}"] = right_names[i]
                else:
                    temp_name_to_result_name[f"{i}_{j}"] = f"{left_names[i]}_{j - 1}_{right_names[i]}"
        if reverse:
            return [
                PermutationChannel(
                    left_nodes=frozenset([temp_name_to_result_name[left_node] for left_node in channel.left_nodes]),
                    right_nodes=frozenset([temp_name_to_result_name[right_node] for right_node in channel.right_nodes]),
                )
                for channel in temp_result_channels
            ]
        else:
            res = [
                PermutationChannel(
                    left_nodes=frozenset([temp_name_to_result_name[left_node] for left_node in channel.right_nodes]),
                    right_nodes=frozenset([temp_name_to_result_name[right_node] for right_node in channel.left_nodes]),
                )
                for channel in temp_result_channels
            ]
            return res[::-1]


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
    pprint.pprint(OddEvenMergeSortNetwork.generate_network(left_nodes, right_nodes))
