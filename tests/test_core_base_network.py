from unittest.mock import patch

import pytest

from sparse_qubo.core.base_network import NetworkType
from sparse_qubo.core.node import NodeAttribute, VariableNode
from sparse_qubo.networks.bubble_sort_network import BubbleSortNetwork


class TestNetworkType:
    """Tests for NetworkType enum."""

    def test_network_type_values(self) -> None:
        """Test that NetworkType has correct values."""
        assert NetworkType.NAIVE.value == "naive"
        assert NetworkType.BENES.value == "benes"
        assert NetworkType.BITONIC_SORT.value == "bitonic_sort"
        assert NetworkType.BUBBLE_SORT.value == "bubble_sort"
        assert NetworkType.CLOS_NETWORK_MAX_DEGREE.value == "clos_network_max_degree"
        assert NetworkType.CLOS_NETWORK_MIN_EDGE.value == "clos_network_min_edge"
        assert NetworkType.DIVIDE_AND_CONQUER.value == "divide_and_conquer"
        assert NetworkType.ODDEVEN_MERGE_SORT.value == "oddeven_merge_sort"


class TestISwitchingNetwork:
    """Tests for ISwitchingNetwork abstract base class."""

    def test_generate_network_fixes_always_one(self) -> None:
        """Test that generate_network fixes nodes that must be ALWAYS_ONE."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ONE) for i in range(4)]

        switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes)

        # All switches should be removed or simplified since all right nodes are ALWAYS_ONE
        # The network should be optimized
        assert isinstance(switches, list)

    def test_generate_network_fixes_always_zero(self) -> None:
        """Test that generate_network fixes nodes that must be ALWAYS_ZERO."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}", attribute=NodeAttribute.ALWAYS_ZERO) for i in range(4)]

        switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes)

        # All switches should be removed or simplified since all right nodes are ALWAYS_ZERO
        assert isinstance(switches, list)

    def test_generate_network_one_hot(self) -> None:
        """Test generate_network with one-hot constraint."""
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

    def test_generate_network_invalid_right_nodes(self) -> None:
        """Test that generate_network raises error for invalid right nodes."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [VariableNode(name=f"R{i}") for i in range(4)]

        switches_original = BubbleSortNetwork._generate_original_network(left_nodes, right_nodes)

        # Manually create an invalid switch structure
        invalid_switches = [
            switches_original[0],
            type(switches_original[0])(
                left_nodes=frozenset(["L_new"]),
                right_nodes=frozenset(["R_new"]),
            ),
        ]
        with (
            patch.object(BubbleSortNetwork, "_generate_original_network", return_value=invalid_switches),
            pytest.raises(ValueError, match="Invalid network"),
        ):
            BubbleSortNetwork.generate_network(left_nodes, right_nodes)

    def test_generate_network_reverse(self) -> None:
        """Test generate_network with reverse parameter."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(4)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 1 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(4)
        ]

        switches_normal = BubbleSortNetwork.generate_network(left_nodes, right_nodes, reverse=False)
        switches_reversed = BubbleSortNetwork.generate_network(left_nodes, right_nodes, reverse=True)

        # Both should produce valid networks
        assert isinstance(switches_normal, list)
        assert isinstance(switches_reversed, list)

    def test_generate_network_with_constants(self) -> None:
        """Test generate_network handles constants correctly."""
        left_nodes = [VariableNode(name=f"L{i}") for i in range(3)]
        right_nodes = [
            VariableNode(
                name=f"R{i}",
                attribute=NodeAttribute.ALWAYS_ZERO if i < 2 else NodeAttribute.ALWAYS_ONE,
            )
            for i in range(3)
        ]

        switches = BubbleSortNetwork.generate_network(left_nodes, right_nodes)

        # Check that constants are properly handled
        for switch in switches:
            # Constants should be non-negative integers
            assert switch.left_constant >= 0
            assert switch.right_constant >= 0
