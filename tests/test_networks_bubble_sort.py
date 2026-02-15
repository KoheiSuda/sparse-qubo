import pytest

from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.bubble_sort_network import BubbleSortNetwork


class TestBubbleSortNetwork:
    """Tests for BubbleSortNetwork."""

    def test_bubble_sort_network_same_length(self) -> None:
        """Test that BubbleSortNetwork requires same length left and right nodes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(4)]

        with pytest.raises(ValueError, match="must have the same length"):
            BubbleSortNetwork._generate_original_network(left_nodes, right_nodes)

    def test_bubble_sort_network_one_hot(self) -> None:
        """Test BubbleSortNetwork with one-hot constraint."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]

        switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes)

        # Check the structure of the network
        c0, c1, c2 = switches[0], switches[1], switches[2]
        # c2 (L0, L1) -> c1
        # c2's output nodes are included in c1's input nodes
        assert c2.right_nodes.issubset(c1.left_nodes)
        # c1 (..., L2) -> c0
        assert c1.right_nodes.issubset(c0.left_nodes)
        # The right side of the last stage is empty
        assert not c0.right_nodes

    def test_bubble_sort_network_all_zero(self) -> None:
        """Test BubbleSortNetwork with all zeros: left nodes are determined to 0, no Switch needed."""
        size = 3
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(size)]

        switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(switches) == 0

    def test_bubble_sort_network_all_one(self) -> None:
        """Test BubbleSortNetwork with all ones: left nodes are determined to 1, no Switch needed."""
        size = 3
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(size)]

        switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes)
        assert len(switches) == 0

    def test_bubble_sort_network_reverse(self) -> None:
        """Test BubbleSortNetwork with reverse parameter."""
        size = 4
        left_nodes = [VariableNode(name=f"L{i}") for i in range(size)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < size - 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(size)
        ]

        switches_normal = BubbleSortNetwork.generate_network(left_nodes, right_nodes, reverse=False)
        switches_reversed = BubbleSortNetwork.generate_network(left_nodes, right_nodes, reverse=True)

        # Reversed should have same number of switches
        assert len(switches_normal) == len(switches_reversed)
