from ..base_network import ISwitchingNetwork
from ..node import NodeAttribute, VariableNode
from ..permutation_channel import PermutationChannel


# Generate a network representing bubble sort
class BubbleSortNetwork(ISwitchingNetwork):
    @classmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[PermutationChannel]:
        left_names = [node.name for node in left_nodes]
        right_names = [node.name for node in right_nodes]
        if len(left_names) != len(right_names):
            raise ValueError("left_nodes and right_nodes must have the same length")
        N: int = len(left_names)

        all_nodes: list[list[str]] = [[] for _ in range(N)]
        for i in range(N):
            all_nodes[i].append(left_names[i])
            for j in range((N - 1 - i) * 2 if i > 0 else N - 2):
                all_nodes[i].append(f"{left_names[i]}_{j}_{right_names[i]}")
            all_nodes[i].append(right_names[i])

        progress: list[int] = [0] * N
        result_channels: list[PermutationChannel] = []
        for i in list(range(1, N)) + list(range(1, N - 1)[::-1]):
            for j in range(0, i, 2):
                k1, k2 = i - j, i - j - 1
                result_channels.append(
                    PermutationChannel(
                        left_nodes=frozenset([all_nodes[k1][progress[k1]], all_nodes[k2][progress[k2]]]),
                        right_nodes=frozenset([
                            all_nodes[k1][progress[k1] + 1],
                            all_nodes[k2][progress[k2] + 1],
                        ]),
                    )
                )
                progress[k1] += 1
                progress[k2] += 1
        return result_channels


if __name__ == "__main__":
    import pprint

    size = 4
    left_nodes: list[VariableNode] = [VariableNode(name=f"L{i}") for i in range(size)]
    right_nodes: list[VariableNode] = [
        VariableNode(
            name=f"R{i}",
            attribute=NodeAttribute.ALWAYS_ZERO if i < size - 1 else NodeAttribute.ALWAYS_ONE,
        )
        for i in range(size)
    ]
    pprint.pprint(BubbleSortNetwork.generate_network(left_nodes, right_nodes))
