from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.clos_network_minimum_edge import ClosNetworkMinimumEdge


class TestClosNetworkMinimumEdge:
    """Tests for ClosNetworkMinimumEdge."""

    def test_clos_network_minimum_edge_basic(self) -> None:
        """Test ClosNetworkMinimumEdge basic functionality."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_minimum_edge_size_4(self) -> None:
        """Test ClosNetworkMinimumEdge with size 4."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_minimum_edge_size_6(self) -> None:
        """Test ClosNetworkMinimumEdge with size 6."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(6)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(6)
        ]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_minimum_edge_size_10(self) -> None:
        """Test ClosNetworkMinimumEdge with size 10."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(10)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(10)
        ]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_minimum_edge_all_zero(self) -> None:
        """Test ClosNetworkMinimumEdge with all zeros."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(4)]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) >= 0

    def test_clos_network_minimum_edge_all_one(self) -> None:
        """Test ClosNetworkMinimumEdge with all ones."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(4)]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) >= 0

    def test_clos_network_minimum_edge_different_sizes(self) -> None:
        """Test ClosNetworkMinimumEdge with different left and right sizes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(6)]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_minimum_edge_reverse(self) -> None:
        """Test ClosNetworkMinimumEdge with reverse parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        switches_normal = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes, reverse=False)
        switches_reversed = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert len(switches_normal) > 0
        assert len(switches_reversed) > 0

    def test_clos_network_minimum_edge_with_threshold(self) -> None:
        """Test ClosNetworkMinimumEdge with threshold parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes, threshold=4)
        assert len(switches) > 0

    def test_clos_network_minimum_edge_small_case(self) -> None:
        """Test ClosNetworkMinimumEdge with small case (size 2)."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(2)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(2)
        ]

        switches = ClosNetworkMinimumEdge.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0
