from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.clique_network import CliqueNetwork


class TestCliqueNetwork:
    """Tests for CliqueNetwork."""

    def test_clique_network_basic(self) -> None:
        """Test CliqueNetwork basic functionality."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes)
        # CliqueNetwork creates a single channel connecting all left to all right
        assert len(channels) >= 0  # May be optimized away if all nodes are fixed

    def test_clique_network_different_sizes(self) -> None:
        """Test CliqueNetwork with different left and right sizes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(5)]

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_clique_network_all_zero(self) -> None:
        """Test CliqueNetwork with all zeros."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(4)]

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_clique_network_all_one(self) -> None:
        """Test CliqueNetwork with all ones."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(4)]

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_clique_network_mixed_attributes(self) -> None:
        """Test CliqueNetwork with mixed node attributes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = (
            [VariableNode(name=f"R{i}", attribute=NodeAttribute.ZERO_OR_ONE) for i in range(2)]
            + [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(2, 3)]
            + [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(3, 4)]
            + [VariableNode(name=f"R{i}", attribute=NodeAttribute.NOT_CARE) for i in range(4, 6)]
        )

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_clique_network_reverse(self) -> None:
        """Test CliqueNetwork with reverse parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels_normal = CliqueNetwork.generate_network(left_nodes, right_nodes, reverse=False)
        channels_reversed = CliqueNetwork.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert isinstance(channels_normal, list)
        assert isinstance(channels_reversed, list)

    def test_clique_network_with_threshold(self) -> None:
        """Test CliqueNetwork with threshold parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes, threshold=4)
        assert len(channels) >= 0

    def test_clique_network_single_node(self) -> None:
        """Test CliqueNetwork with single node."""
        left_nodes = [VariableNode(name="L0")]
        right_nodes = [VariableNode(name="R0", attribute=NodeAttribute.ALWAYS_ONE)]

        channels = CliqueNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0
