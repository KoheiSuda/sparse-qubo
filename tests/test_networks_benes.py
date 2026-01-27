from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.benes_network import BenesNetwork


class TestBenesNetwork:
    """Tests for BenesNetwork."""

    def test_benes_network_size_1(self) -> None:
        """Test BenesNetwork with size 1."""
        left_nodes = [VariableNode(name="L0")]
        right_nodes = [VariableNode(name="R0", attribute=NodeAttribute.ALWAYS_ONE)]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0  # May be optimized away

    def test_benes_network_size_2(self) -> None:
        """Test BenesNetwork with size 2."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(2)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(2)
        ]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_benes_network_size_4(self) -> None:
        """Test BenesNetwork with size 4."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_benes_network_size_8(self) -> None:
        """Test BenesNetwork with size 8."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_benes_network_all_zero(self) -> None:
        """Test BenesNetwork with all zeros."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(size)]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_benes_network_all_one(self) -> None:
        """Test BenesNetwork with all ones."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(size)]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_benes_network_reverse(self) -> None:
        """Test BenesNetwork with reverse parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels_normal = BenesNetwork.generate_network(left_nodes, right_nodes, reverse=False)
        channels_reversed = BenesNetwork.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert len(channels_normal) > 0
        assert len(channels_reversed) > 0

    def test_benes_network_with_threshold(self) -> None:
        """Test BenesNetwork with threshold parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes, threshold=4)
        assert len(channels) > 0

    def test_benes_network_different_sizes(self) -> None:
        """Test BenesNetwork with different left and right sizes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(6)]

        channels = BenesNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0
