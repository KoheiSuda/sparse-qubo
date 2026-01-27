import pydantic
import pytest

from sparse_qubo.core.node import NodeAttribute, VariableNode


class TestNodeAttribute:
    """Tests for NodeAttribute enum."""

    def test_node_attribute_values(self) -> None:
        """Test that NodeAttribute has correct values."""
        assert NodeAttribute.ZERO_OR_ONE.value == "ZERO_OR_ONE"
        assert NodeAttribute.ALWAYS_ZERO.value == "ALWAYS_ZERO"
        assert NodeAttribute.ALWAYS_ONE.value == "ALWAYS_ONE"
        assert NodeAttribute.NOT_CARE.value == "NOT_CARE"

    def test_node_attribute_enum_membership(self) -> None:
        """Test NodeAttribute enum membership."""
        assert NodeAttribute.ZERO_OR_ONE in NodeAttribute
        assert NodeAttribute.ALWAYS_ZERO in NodeAttribute
        assert NodeAttribute.ALWAYS_ONE in NodeAttribute
        assert NodeAttribute.NOT_CARE in NodeAttribute


class TestVariableNode:
    """Tests for VariableNode model."""

    def test_variable_node_creation(self) -> None:
        """Test creating a VariableNode with default attribute."""
        node = VariableNode(name="x1")
        assert node.name == "x1"
        assert node.attribute == NodeAttribute.ZERO_OR_ONE

    def test_variable_node_with_custom_attribute(self) -> None:
        """Test creating a VariableNode with custom attribute."""
        node = VariableNode(name="x1", attribute=NodeAttribute.ALWAYS_ONE)
        assert node.name == "x1"
        assert node.attribute == NodeAttribute.ALWAYS_ONE

    def test_variable_node_name_frozen(self) -> None:
        """Test that VariableNode name is frozen."""
        node = VariableNode(name="x1")
        with pytest.raises(pydantic.ValidationError):
            node.name = "x2"

    @pytest.mark.parametrize("attr", list(NodeAttribute))
    def test_variable_node_all_attributes(self, attr: NodeAttribute) -> None:
        """Test creating VariableNode with all attribute types."""
        node = VariableNode(name=f"node_{attr.value}", attribute=attr)
        assert node.attribute == attr
