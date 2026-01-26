import dimod
import dimod.variables

from ..core.constraint import ConstraintType, get_constraint_qubo


def naive_constraint(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> dimod.BinaryQuadraticModel:
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    size = len(variables)
    if constraint_type == ConstraintType.ONE_HOT:
        bqm.add_linear_equality_constraint(
            [(str(v), 1) for v in variables],
            1,
            -1,
        )
        return bqm
    if constraint_type == ConstraintType.EQUAL_TO:
        if not (c1 is not None and 0 <= c1 <= size):
            raise ValueError("c1 must be between 0 and size")
        bqm.add_linear_equality_constraint(
            [(str(v), 1) for v in variables],
            1,
            -c1,
        )
        return bqm
    else:
        raise NotImplementedError


def constraint(
    variables: dimod.variables.Variables,
    constraint_type: ConstraintType,
    network_name: str = "divide_and_conquer",
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
) -> dimod.BinaryQuadraticModel:
    if network_name == "divide_and_conquer":
        variable_names = [str(v) for v in variables]
        qubo = get_constraint_qubo(variable_names, constraint_type, network_name, c1, c2, threshold)
        bqm = dimod.BinaryQuadraticModel(
            qubo.linear,
            qubo.quadratic,
            qubo.constant,
            dimod.BINARY,
        )
        return bqm
    elif network_name == "naive":
        bqm = naive_constraint(variables, constraint_type, c1, c2)
        return bqm
    else:
        raise NotImplementedError
