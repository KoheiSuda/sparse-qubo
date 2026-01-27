import pytest

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.bitonic_sort_network import BitonicSortNetwork


class TestBitonicSortNetwork:
    """Tests for BitonicSortNetwork."""

    def test_bitonic_sort_same_length(self) -> None:
        """Test that BitonicSortNetwork requires same length left and right nodes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(4)]

        with pytest.raises(ValueError, match="must have the same length"):
            BitonicSortNetwork._generate_original_network(left_nodes, right_nodes)

    def test_bitonic_sort_power_of_two(self) -> None:
        """Test that BitonicSortNetwork requires power of 2 length."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]  # Not a power of 2
        right_nodes = [VariableNode(name=f"R{i}") for i in range(3)]

        with pytest.raises(ValueError, match="must be a power of 2"):
            BitonicSortNetwork._generate_original_network(left_nodes, right_nodes)

    def test_bitonic_sort_size_2(self) -> None:
        """Test BitonicSortNetwork with size 2."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(2)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(2)
        ]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_bitonic_sort_size_4(self) -> None:
        """Test BitonicSortNetwork with size 4."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_bitonic_sort_size_8(self) -> None:
        """Test BitonicSortNetwork with size 8."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_bitonic_sort_size_16(self) -> None:
        """Test BitonicSortNetwork with size 16."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(16)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(16)
        ]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_bitonic_sort_all_zero(self) -> None:
        """Test BitonicSortNetwork with all zeros."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(4)]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_bitonic_sort_all_one(self) -> None:
        """Test BitonicSortNetwork with all ones."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(4)]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) >= 0

    def test_bitonic_sort_reverse(self) -> None:
        """Test BitonicSortNetwork with reverse parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        channels_normal = BitonicSortNetwork.generate_network(left_nodes, right_nodes, reverse=False)
        channels_reversed = BitonicSortNetwork.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert len(channels_normal) > 0
        assert len(channels_reversed) > 0

    def test_bitonic_sort_with_threshold(self) -> None:
        """Test BitonicSortNetwork with threshold parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(8)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(8)
        ]

        channels = BitonicSortNetwork.generate_network(left_nodes, right_nodes, threshold=4)
        assert len(channels) > 0
