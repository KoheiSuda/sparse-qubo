from ..base_network import ISwitchingNetwork
from ..node import NodeAttribute, VariableNode
from ..permutation_channel import PermutationChannel


# Generate a network that connects left and right in an all-to-all manner
class CliqueNetwork(ISwitchingNetwork):
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
        return [
            PermutationChannel(
                left_nodes=frozenset(left_names),
                right_nodes=frozenset(right_names),
            )
        ]


if __name__ == "__main__":
    left_nodes = [VariableNode(name=f"x_{i}", attribute=NodeAttribute.ZERO_OR_ONE) for i in range(4)]
    right_nodes = (
        [VariableNode(name=f"y_{i}", attribute=NodeAttribute.ZERO_OR_ONE) for i in range(4)]
        + [VariableNode(name=f"y_{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(4, 5)]
        + [VariableNode(name=f"y_{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(5, 8)]
        + [VariableNode(name=f"y_{i}", attribute=NodeAttribute.NOT_CARE) for i in range(8, 10)]
    )
    print(CliqueNetwork.generate_network(left_nodes, right_nodes))
