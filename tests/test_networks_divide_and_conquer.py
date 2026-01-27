import pytest

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.divide_and_conquer_network import DivideAndConquerNetwork


class TestDivideAndConquerNetwork:
    """Tests for DivideAndConquerNetwork."""

    def test_divide_and_conquer_same_length(self) -> None:
        """Test that DivideAndConquerNetwork requires same length left and right nodes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(4)]

        with pytest.raises(ValueError, match="must have the same length"):
            DivideAndConquerNetwork._generate_original_network(left_nodes, right_nodes)

    def test_divide_and_conquer_all_zero(self) -> None:
        """Test DivideAndConquerNetwork with all zeros."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(size)]

        channels = DivideAndConquerNetwork.generate_network(left_nodes, right_nodes)
        # Should have direct connections
        assert len(channels) == size

    def test_divide_and_conquer_all_one(self) -> None:
        """Test DivideAndConquerNetwork with all ones."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(size)]

        channels = DivideAndConquerNetwork.generate_network(left_nodes, right_nodes)
        # Should have direct connections
        assert len(channels) == size

    def test_divide_and_conquer_one_hot(self) -> None:
        """Test DivideAndConquerNetwork with one-hot constraint."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]

        channels = DivideAndConquerNetwork.generate_network(left_nodes, right_nodes)
        assert len(channels) > 0

    def test_divide_and_conquer_with_threshold(self) -> None:
        """Test DivideAndConquerNetwork with threshold parameter."""
        size = 8
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size // 2 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]

        channels_with_threshold = DivideAndConquerNetwork.generate_network(left_nodes, right_nodes, threshold=4)
        channels_without_threshold = DivideAndConquerNetwork.generate_network(left_nodes, right_nodes)

        # Both should produce valid networks
        assert len(channels_with_threshold) > 0
        assert len(channels_without_threshold) > 0

    def test_divide_and_conquer_invalid_left_nodes(self) -> None:
        """Test DivideAndConquerNetwork raises error for invalid left nodes."""
        size = 4
        left_nodes = [
            VariableNode(name=f"L{i}", attribute=NodeAttribute.ALWAYS_ONE if i == 0 else NodeAttribute.ZERO_OR_ONE)
            for i in range(size)
        ]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]

        with pytest.raises(ValueError, match="All left_nodes must have ZERO_OR_ONE attribute"):
            DivideAndConquerNetwork._generate_original_network(left_nodes, right_nodes)

    def test_divide_and_conquer_invalid_right_nodes(self) -> None:
        """Test DivideAndConquerNetwork raises error for invalid right nodes."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ZERO_OR_ONE) for i in range(size)]

        with pytest.raises(ValueError, match="ZERO_OR_ONE nodes are not supported"):
            DivideAndConquerNetwork._generate_original_network(left_nodes, right_nodes)

    def test_divide_and_conquer_not_care_not_supported(self) -> None:
        """Test DivideAndConquerNetwork raises error for NOT_CARE nodes."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.NOT_CARE) for i in range(size)]

        with pytest.raises(ValueError, match="NOT_CARE nodes are not supported"):
            DivideAndConquerNetwork._generate_original_network(left_nodes, right_nodes)
