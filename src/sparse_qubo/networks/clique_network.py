"""Implementation of clique network: single all-to-all switch between left and right variables."""

from sparse_qubo.core.network import ISwitchingNetwork
from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.core.switch import Switch


class CliqueNetwork(ISwitchingNetwork):
    """Single switch connecting all left variables to all right variables (clique)."""

    @classmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[Switch]:
        """Return a single Switch with all left and all right variables."""
        left_names = [node.name for node in left_nodes]
        right_names = [node.name for node in right_nodes]
        return [
            Switch(
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
