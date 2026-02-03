"""D-Wave/dimod integration: build dimod BQMs from constraint types and network types.

Use create_constraint_dwave (from sparse_qubo) or constraint() from this module
to obtain a BinaryQuadraticModel for use with D-Wave samplers or dimod.
"""

import dimod
import dimod.variables

from sparse_qubo.core.constraint import ConstraintType, get_constraint_qubo
from sparse_qubo.core.network import NetworkType


def naive_constraint(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> dimod.BinaryQuadraticModel:
    """Encode the constraint as a single linear equality/inequality (no switching network)."""
    size = len(variables)
    terms = [(str(v), 1) for v in variables]
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    lagrange_multiplier = 1
    match constraint_type:
        case ConstraintType.ONE_HOT:
            bqm.add_linear_equality_constraint(terms, lagrange_multiplier, -1)
        case ConstraintType.EQUAL_TO:
            if not (c1 is not None and 0 <= c1 <= size):
                raise ValueError("c1 must be between 0 and size")
            bqm.add_linear_equality_constraint(terms, lagrange_multiplier, -c1)
        case ConstraintType.LESS_EQUAL:
            if not (c1 is not None and 0 <= c1 <= size):
                raise ValueError("c1 must be between 0 and size")
            bqm.add_linear_inequality_constraint(terms, lagrange_multiplier, label="s", ub=c1)
        case ConstraintType.GREATER_EQUAL:
            if not (c1 is not None and 0 <= c1 <= size):
                raise ValueError("c1 must be between 0 and size")
            bqm.add_linear_inequality_constraint(terms, lagrange_multiplier, label="s", lb=c1, ub=size)
        case ConstraintType.CLAMP:
            if not (c1 is not None and c2 is not None and 0 <= c1 <= c2 <= size):
                raise ValueError("c1 and c2 must be between 0 and size")
            bqm.add_linear_inequality_constraint(terms, lagrange_multiplier, label="s", lb=c1, ub=c2)
        case _:
            raise NotImplementedError

    return bqm


def constraint(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    network_type: NetworkType = NetworkType.DIVIDE_AND_CONQUER,
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
) -> dimod.BinaryQuadraticModel:
    """Build a dimod BQM for the given constraint using the specified network type (or NAIVE)."""
    if network_type == NetworkType.NAIVE:
        bqm = naive_constraint(variables, constraint_type, c1, c2)
        return bqm

    variable_names = [str(v) for v in variables]
    qubo = get_constraint_qubo(variable_names, constraint_type, network_type, c1, c2, threshold)
    bqm = dimod.BinaryQuadraticModel(
        qubo.linear,
        qubo.quadratic,
        qubo.constant,
        dimod.BINARY,
    )
    return bqm
