import pytest

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.clos_network_max_degree import ClosNetworkWithMaxDegree


class TestClosNetworkWithMaxDegree:
    """Tests for ClosNetworkWithMaxDegree."""

    def test_clos_network_max_degree_basic(self) -> None:
        """Test ClosNetworkWithMaxDegree basic functionality."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(7)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(7)
        ]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_max_degree_size_4(self) -> None:
        """Test ClosNetworkWithMaxDegree with size 4."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_max_degree_size_8(self) -> None:
        """Test ClosNetworkWithMaxDegree with size 8."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_max_degree_all_zero(self) -> None:
        """Test ClosNetworkWithMaxDegree with all zeros."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(4)]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
        assert len(switches) >= 0

    def test_clos_network_max_degree_all_one(self) -> None:
        """Test ClosNetworkWithMaxDegree with all ones."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(4)]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
        assert len(switches) >= 0

    def test_clos_network_max_degree_different_sizes(self) -> None:
        """Test ClosNetworkWithMaxDegree with different left and right sizes."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(6)]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
        assert len(switches) > 0

    def test_clos_network_max_degree_reverse(self) -> None:
        """Test ClosNetworkWithMaxDegree with reverse parameter."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        switches_normal = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes, reverse=False)
        switches_reversed = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert len(switches_normal) > 0
        assert len(switches_reversed) > 0

    def test_clos_network_max_degree_with_threshold(self) -> None:
        """Test ClosNetworkWithMaxDegree with threshold parameter."""
        ClosNetworkWithMaxDegree.reset_max_degree(5)
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes, threshold=4)
        assert len(switches) > 0

    def test_reset_max_degree(self) -> None:
        """Test reset_max_degree method."""
        ClosNetworkWithMaxDegree.reset_max_degree(3)
        assert ClosNetworkWithMaxDegree.max_degree == 3

        ClosNetworkWithMaxDegree.reset_max_degree(7)
        assert ClosNetworkWithMaxDegree.max_degree == 7

    def test_reset_max_degree_invalid(self) -> None:
        """Test reset_max_degree raises error for invalid values."""
        with pytest.raises(ValueError, match="must be greater than or equal to 2"):
            ClosNetworkWithMaxDegree.reset_max_degree(1)

        with pytest.raises(ValueError, match="must be greater than or equal to 2"):
            ClosNetworkWithMaxDegree.reset_max_degree(0)

    def test_clos_network_max_degree_different_max_degrees(self) -> None:
        """Test ClosNetworkWithMaxDegree with different max_degree values."""
        for max_degree in [3, 4, 5, 6]:
            ClosNetworkWithMaxDegree.reset_max_degree(max_degree)
            left_nodes = [VariableNode(name=f"L{i}") for i in range(7)]
            right_nodes = [
                VariableNode(
                    name=f"R{i}",
                    attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
                )
                for i in range(7)
            ]

            switches = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
            assert len(switches) > 0
