import amplify

from sparse_qubo.core.constraint import ConstraintType, get_constraint_qubo
from sparse_qubo.core.permutation_channel import QUBO


def naive_constraint(
    variables: list[amplify.Variable],
    constraint_type: ConstraintType,
    c1: int | None = None,
    c2: int | None = None,
) -> amplify.Model:
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


def generate_amplify_model(variables: list[amplify.Variable], qubo: QUBO) -> amplify.Model:
    poly_map = {v.name: amplify.Poly(v) for v in variables}
    objectives = amplify.Poly(0)

    for (v1, v2), coef in qubo.quadratic.items():
        objectives += coef * poly_map[v1] * poly_map[v2]
    for v, coef in qubo.linear.items():
        objectives += coef * poly_map[v]
    objectives += qubo.constant

    return amplify.Model(objectives)


def constraint(
    variables: list[amplify.Variable],
    constraint_type: ConstraintType,
    network_name: str = "divide_and_conquer",
    c1: int | None = None,
    c2: int | None = None,
    threshold: int | None = None,
) -> amplify.Model:
    if network_name == "naive":
        model = naive_constraint(variables, constraint_type, c1, c2)
        return model

    variable_names = [v.name for v in variables]
    if network_name == "divide_and_conquer":
        qubo = get_constraint_qubo(variable_names, constraint_type, network_name, c1, c2, threshold)
        model = generate_amplify_model(variables, qubo)
        return model

    else:
        raise NotImplementedError
