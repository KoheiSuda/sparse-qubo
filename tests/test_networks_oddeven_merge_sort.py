import pytest

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.oddeven_merge_sort_network import OddEvenMergeSortNetwork


class TestOddEvenMergeSortNetwork:
    """Tests for OddEvenMergeSortNetwork."""

    def test_oddeven_merge_sort_same_length(self) -> None:
        """Test that OddEvenMergeSortNetwork requires same length left and right nodes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(4)]

        with pytest.raises(ValueError, match="must have the same length"):
            OddEvenMergeSortNetwork._generate_original_network(left_nodes, right_nodes)

    def test_oddeven_merge_sort_power_of_two(self) -> None:
        """Test that OddEvenMergeSortNetwork requires power of 2 length."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]  # Not a power of 2
        right_nodes = [VariableNode(name=f"R{i}") for i in range(3)]

        with pytest.raises(ValueError, match="must be a power of 2"):
            OddEvenMergeSortNetwork._generate_original_network(left_nodes, right_nodes)

    def test_oddeven_merge_sort_size_2(self) -> None:
        """Test OddEvenMergeSortNetwork with size 2."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(2)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(2)
        ]

        channels = OddEvenMergeSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_oddeven_merge_sort_size_4(self) -> None:
        """Test OddEvenMergeSortNetwork with size 4."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels = OddEvenMergeSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_oddeven_merge_sort_size_8(self) -> None:
        """Test OddEvenMergeSortNetwork with size 8."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        channels = OddEvenMergeSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_oddeven_merge_sort_reverse(self) -> None:
        """Test OddEvenMergeSortNetwork with reverse parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels_normal = OddEvenMergeSortNetwork.generate_network(left_nodes, right_nodes, reverse=False)
        channels_reversed = OddEvenMergeSortNetwork.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert len(channels_normal) > 0
        assert len(channels_reversed) > 0
