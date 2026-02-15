"""Fixstars Amplify integration: build Amplify models from constraint types and network types.

Use create_constraint_amplify (from sparse_qubo) or constraint() from this module
to obtain an amplify.Model for use with Amplify solvers.
"""

import amplify

from sparse_qubo.core.constraint import ConstraintType, get_constraint_switches
from sparse_qubo.core.network import NetworkType
from sparse_qubo.core.switch import Switch, get_variables_from_switches


def naive_constraint(
    variables: list[amplify.Variable],
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> amplify.Model:
    """Encode the constraint using Amplify's built-in one_hot/equal_to/less_equal/etc. (no switching network)."""
    size = len(variables)
    sum_poly: amplify.Poly = sum([amplify.Poly(v) for v in variables], amplify.Poly(0))
    match constraint_type:
        case ConstraintType.ONE_HOT:
            constraint = amplify.one_hot(sum_poly)
        case ConstraintType.EQUAL_TO:
            if not (c1 is not None and 0 <= c1 <= size):
                raise ValueError("c1 must be between 0 and size")
            constraint = amplify.equal_to(sum_poly, c1)
        case ConstraintType.LESS_EQUAL:
            if not (c1 is not None and 0 <= c1 <= size):
                raise ValueError("c1 must be between 0 and size")
            constraint = amplify.less_equal(sum_poly, c1)
        case ConstraintType.GREATER_EQUAL:
            if not (c1 is not None and 0 <= c1 <= size):
                raise ValueError("c1 must be between 0 and size")
            constraint = amplify.greater_equal(sum_poly, c1)
        case ConstraintType.CLAMP:
            if not (c1 is not None and c2 is not None and 0 <= c1 <= c2 <= size):
                raise ValueError("c1 and c2 must be between 0 and size")
            constraint = amplify.clamp(sum_poly, (c1, c2))
        case _:
            raise NotImplementedError

    return amplify.Model(constraint)


def generate_amplify_model(gen: amplify.VariableGenerator, switches: list[Switch]) -> amplify.Model:
    """Build an Amplify model from a list of Amplify variables and a list of Switches."""
    all_vars = [gen.scalar("Binary", name=name).as_variable() for name in get_variables_from_switches(switches)]
    vars_dict = {v.name: v for v in all_vars}
    model = amplify.Model()
    for switch in switches:
        model += amplify.equal_to(
            sum([amplify.Poly(vars_dict[v]) for v in switch.left_nodes])
            - sum([amplify.Poly(vars_dict[v]) for v in switch.right_nodes]),
            switch.right_constant - switch.left_constant,
        )
    return model


def constraint(
    gen: amplify.VariableGenerator,
    variables: list[amplify.Variable],
    constraint_type: ConstraintType,
    network_type: NetworkType = NetworkType.DIVIDE_AND_CONQUER,
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
) -> amplify.Model:
    """Build an Amplify model for the given constraint using the specified network type (or NAIVE)."""
    if network_type == NetworkType.NAIVE:
        return naive_constraint(variables, constraint_type, c1, c2)

    variable_names = [v.name for v in variables]
    switches = get_constraint_switches(variable_names, constraint_type, network_type, c1, c2, threshold)
    return generate_amplify_model(gen, switches)
