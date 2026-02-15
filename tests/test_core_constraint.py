from typing import cast

import pytest

from sparse_qubo.core.constraint import ConstraintType, get_constraint_switches, get_initial_nodes
from sparse_qubo.core.network import NetworkType
from sparse_qubo.core.node import NodeAttribute
from sparse_qubo.core.switch import switches_to_qubo


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_constraint_type_values(self) -> None:
        """Test that ConstraintType has correct values."""
        assert ConstraintType.ONE_HOT.value == "one_hot"
        assert ConstraintType.EQUAL_TO.value == "equal_to"
        assert ConstraintType.LESS_EQUAL.value == "less_equal"
        assert ConstraintType.GREATER_EQUAL.value == "greater_equal"
        assert ConstraintType.CLAMP.value == "clamp"


class TestGetInitialNodes:
    """Tests for get_initial_nodes function."""

    def test_one_hot_constraint(self) -> None:
        """Test get_initial_nodes for ONE_HOT constraint."""
        variables = ["x0", "x1", "x2"]
        left_nodes, right_nodes = get_initial_nodes(variables, ConstraintType.ONE_HOT)

        # Left nodes: all ZERO_OR_ONE
        assert len(left_nodes) == 3
        assert all(node.attribute == NodeAttribute.ZERO_OR_ONE for node in left_nodes)

        # Right nodes: last one is ALWAYS_ONE, others are ALWAYS_ZERO
        assert len(right_nodes) == 3
        assert right_nodes[0].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[1].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[2].attribute == NodeAttribute.ALWAYS_ONE

    def test_one_hot_with_exponentiation(self) -> None:
        """Test get_initial_nodes for ONE_HOT with exponentiation."""
        variables = ["x0", "x1", "x2"]
        left_nodes, right_nodes = get_initial_nodes(variables, ConstraintType.ONE_HOT, exponentiation=True)

        # Size should be padded to power of 2 (4 in this case)
        assert len(left_nodes) == 4
        assert len(right_nodes) == 4

        # First node should be ALWAYS_ZERO (padding)
        assert left_nodes[0].attribute == NodeAttribute.ALWAYS_ZERO

    def test_equal_to_constraint(self) -> None:
        """Test get_initial_nodes for EQUAL_TO constraint."""
        variables = ["x0", "x1", "x2", "x3"]
        _, right_nodes = get_initial_nodes(variables, ConstraintType.EQUAL_TO, c1=2)

        # Right nodes: last 2 should be ALWAYS_ONE, first 2 should be ALWAYS_ZERO
        assert right_nodes[0].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[1].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[2].attribute == NodeAttribute.ALWAYS_ONE
        assert right_nodes[3].attribute == NodeAttribute.ALWAYS_ONE

    def test_equal_to_constraint_invalid_c1(self) -> None:
        """Test get_initial_nodes raises error for invalid c1 in EQUAL_TO."""
        variables = ["x0", "x1", "x2"]
        with pytest.raises(ValueError, match="c1 must be between"):
            get_initial_nodes(variables, ConstraintType.EQUAL_TO, c1=5)  # c1 > len(variables)

    def test_less_equal_constraint(self) -> None:
        """Test get_initial_nodes for LESS_EQUAL constraint."""
        variables = ["x0", "x1", "x2", "x3"]
        _, right_nodes = get_initial_nodes(variables, ConstraintType.LESS_EQUAL, c1=2)

        # Right nodes: last 2 should be NOT_CARE, first 2 should be ALWAYS_ZERO
        assert right_nodes[0].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[1].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[2].attribute == NodeAttribute.NOT_CARE
        assert right_nodes[3].attribute == NodeAttribute.NOT_CARE

    def test_greater_equal_constraint(self) -> None:
        """Test get_initial_nodes for GREATER_EQUAL constraint."""
        variables = ["x0", "x1", "x2", "x3"]
        _, right_nodes = get_initial_nodes(variables, ConstraintType.GREATER_EQUAL, c1=2)

        # Right nodes: last 2 should be ALWAYS_ONE, first 2 should be NOT_CARE
        assert right_nodes[0].attribute == NodeAttribute.NOT_CARE
        assert right_nodes[1].attribute == NodeAttribute.NOT_CARE
        assert right_nodes[2].attribute == NodeAttribute.ALWAYS_ONE
        assert right_nodes[3].attribute == NodeAttribute.ALWAYS_ONE

    def test_clamp_constraint(self) -> None:
        """Test get_initial_nodes for CLAMP constraint."""
        variables = ["x0", "x1", "x2", "x3", "x4"]
        _, right_nodes = get_initial_nodes(variables, ConstraintType.CLAMP, c1=1, c2=3)

        # Right nodes: indices 0-1 should be ALWAYS_ZERO, 2-3 should be NOT_CARE, 4 should be ALWAYS_ONE
        assert right_nodes[0].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[1].attribute == NodeAttribute.ALWAYS_ZERO
        assert right_nodes[2].attribute == NodeAttribute.NOT_CARE
        assert right_nodes[3].attribute == NodeAttribute.NOT_CARE
        assert right_nodes[4].attribute == NodeAttribute.ALWAYS_ONE

    def test_clamp_constraint_invalid_range(self) -> None:
        """Test get_initial_nodes raises error for invalid CLAMP range."""
        variables = ["x0", "x1", "x2"]
        with pytest.raises(ValueError, match="c1 and c2 must be valid range"):
            get_initial_nodes(variables, ConstraintType.CLAMP, c1=2, c2=1)  # c1 > c2

    def test_unsupported_constraint_type(self) -> None:
        """Test get_initial_nodes raises error for unsupported constraint type."""
        variables = ["x0", "x1"]
        with pytest.raises(NotImplementedError):
            get_initial_nodes(variables, cast(ConstraintType, "unsupported"))


class TestGetConstraintSwitchesAndQUBO:
    """Tests for get_constraint_switches and switches_to_qubo (replacing get_constraint_qubo)."""

    def test_one_hot_divide_and_conquer(self) -> None:
        """Test get_constraint_switches + switches_to_qubo for ONE_HOT with DIVIDE_AND_CONQUER."""
        variables = ["x0", "x1", "x2"]
        switches = get_constraint_switches(variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0
        assert isinstance(qubo.quadratic, dict)
        assert isinstance(qubo.linear, dict)

    def test_equal_to_bubble_sort(self) -> None:
        """Test get_constraint_switches + switches_to_qubo for EQUAL_TO with BUBBLE_SORT."""
        variables = ["x0", "x1", "x2", "x3"]
        switches = get_constraint_switches(variables, ConstraintType.EQUAL_TO, NetworkType.BUBBLE_SORT, c1=2)
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_less_equal_clos_network(self) -> None:
        """Test get_constraint_switches + switches_to_qubo for LESS_EQUAL with CLOS_NETWORK_MAX_DEGREE."""
        variables = ["x0", "x1", "x2", "x3", "x4"]
        switches = get_constraint_switches(
            variables, ConstraintType.LESS_EQUAL, NetworkType.CLOS_NETWORK_MAX_DEGREE, c1=3
        )
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_greater_equal_bitonic_sort(self) -> None:
        """Test get_constraint_switches + switches_to_qubo for GREATER_EQUAL with BITONIC_SORT."""
        variables = ["x0", "x1", "x2", "x3"]
        switches = get_constraint_switches(variables, ConstraintType.GREATER_EQUAL, NetworkType.BITONIC_SORT, c1=1)
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_clamp_oddeven_merge_sort(self) -> None:
        """Test get_constraint_switches + switches_to_qubo for CLAMP with ODDEVEN_MERGE_SORT."""
        variables = ["x0", "x1", "x2", "x3"]
        switches = get_constraint_switches(variables, ConstraintType.CLAMP, NetworkType.ODDEVEN_MERGE_SORT, c1=1, c2=2)
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_benes_network(self) -> None:
        """Test get_constraint_switches + switches_to_qubo with BENES network."""
        variables = ["x0", "x1", "x2", "x3"]
        switches = get_constraint_switches(variables, ConstraintType.ONE_HOT, NetworkType.BENES)
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_unsupported_network_type(self) -> None:
        """Test get_constraint_switches raises error for unsupported network type."""
        variables = ["x0", "x1"]
        with pytest.raises(NotImplementedError):
            get_constraint_switches(variables, ConstraintType.ONE_HOT, cast(NetworkType, "unsupported"))

    def test_with_threshold(self) -> None:
        """Test get_constraint_switches + switches_to_qubo with threshold parameter."""
        variables = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
        switches = get_constraint_switches(
            variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER, threshold=4
        )
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_with_reverse(self) -> None:
        """Test get_constraint_switches + switches_to_qubo with reverse parameter."""
        variables = ["x0", "x1", "x2", "x3"]
        switches = get_constraint_switches(
            variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER, reverse=True
        )
        qubo = switches_to_qubo(switches)

        assert isinstance(qubo.variables, frozenset)
        assert len(qubo.variables) > 0

    def test_variable_prefix_avoids_collision(self) -> None:
        """Internally assigned prefixes keep auxiliary variable names disjoint across calls."""
        from sparse_qubo.core.constraint import reset_constraint_prefix_counter

        reset_constraint_prefix_counter()
        switches1 = get_constraint_switches(["a", "b", "c"], ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)
        switches2 = get_constraint_switches(["x", "y", "z"], ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)
        qubo1 = switches_to_qubo(switches1)
        qubo2 = switches_to_qubo(switches2)
        # User variables differ; internal prefix makes auxiliary names disjoint
        assert qubo1.variables.isdisjoint(qubo2.variables), "auxiliary variables should not collide"
