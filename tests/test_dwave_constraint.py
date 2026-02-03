from typing import TypedDict

import dimod
import dimod.variables
import pytest

from sparse_qubo.core.base_network import NetworkType
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.dwave.constraint import constraint, naive_constraint


class TestNaiveConstraint:
    """Tests for naive_constraint function."""

    def test_naive_one_hot(self) -> None:
        """Test naive_constraint for ONE_HOT."""
        variables = dimod.variables.Variables(["x0", "x1", "x2"])
        bqm = naive_constraint(variables, ConstraintType.ONE_HOT)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY

    def test_naive_equal_to(self) -> None:
        """Test naive_constraint for EQUAL_TO."""
        variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])
        bqm = naive_constraint(variables, ConstraintType.EQUAL_TO, c1=2)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY

    def test_naive_equal_to_invalid_c1(self) -> None:
        """Test naive_constraint raises error for invalid c1 in EQUAL_TO."""
        variables = dimod.variables.Variables(["x0", "x1", "x2"])
        with pytest.raises(ValueError, match="c1 must be between"):
            naive_constraint(variables, ConstraintType.EQUAL_TO, c1=5)

    def test_naive_less_equal(self) -> None:
        """Test naive_constraint for LESS_EQUAL."""
        variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])
        bqm = naive_constraint(variables, ConstraintType.LESS_EQUAL, c1=2)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY

    def test_naive_greater_equal(self) -> None:
        """Test naive_constraint for GREATER_EQUAL."""
        variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])
        bqm = naive_constraint(variables, ConstraintType.GREATER_EQUAL, c1=2)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY

    def test_naive_clamp(self) -> None:
        """Test naive_constraint for CLAMP."""
        variables = dimod.variables.Variables(["x0", "x1", "x2", "x3", "x4"])
        bqm = naive_constraint(variables, ConstraintType.CLAMP, c1=1, c2=3)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY

    def test_naive_clamp_invalid_range(self) -> None:
        """Test naive_constraint raises error for invalid CLAMP range."""
        variables = dimod.variables.Variables(["x0", "x1", "x2"])
        with pytest.raises(ValueError, match="c1 and c2 must be between"):
            naive_constraint(variables, ConstraintType.CLAMP, c1=2, c2=1)


class _ConstraintTestKwargs(TypedDict, total=False):
    """Keyword args passed to constraint() in tests (c1, c2, threshold only)."""

    c1: int
    c2: int
    threshold: int


class TestConstraint:
    """Tests for constraint function."""

    @pytest.mark.parametrize("network_type", list(NetworkType))
    @pytest.mark.parametrize(
        "constraint_type, kwargs",
        [
            (ConstraintType.ONE_HOT, {}),
            (ConstraintType.EQUAL_TO, {"c1": 4}),
            (ConstraintType.LESS_EQUAL, {"c1": 4}),
            (ConstraintType.GREATER_EQUAL, {"c1": 4}),
            (ConstraintType.CLAMP, {"c1": 2, "c2": 6}),
        ],
    )
    def test_constraint(
        self, network_type: NetworkType, constraint_type: ConstraintType, kwargs: _ConstraintTestKwargs
    ) -> None:
        """Test all combinations of NetworkType and ConstraintType."""
        variables = dimod.variables.Variables([f"x{i}" for i in range(8)])

        if network_type == NetworkType.DIVIDE_AND_CONQUER and constraint_type in {
            ConstraintType.LESS_EQUAL,
            ConstraintType.GREATER_EQUAL,
            ConstraintType.CLAMP,
        }:
            pytest.skip(f"{network_type.name} does not support {constraint_type.name} yet (NOT_CARE nodes)")

        bqm = constraint(variables, constraint_type, network_type, **kwargs)

        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY

    def test_constraint_with_threshold(self) -> None:
        """Test constraint with threshold parameter."""
        variables = dimod.variables.Variables([f"x{i}" for i in range(8)])
        bqm = constraint(variables, ConstraintType.EQUAL_TO, NetworkType.DIVIDE_AND_CONQUER, c1=4, threshold=2)
        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert bqm.vartype == dimod.BINARY
