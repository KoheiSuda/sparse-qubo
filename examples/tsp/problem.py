import itertools

import dimod
import dimod.variables
import numpy as np

import sparse_qubo


def create_tsp_problem(
    num_points: int, density: float = 1.0, max_distance: int = 100, seed: int | None = None
) -> np.ndarray:
    assert num_points >= 2
    assert 0.0 < density <= 1.0

    np.random.seed(seed)

    dist_matrix = np.full((num_points, num_points), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    possible_edges = list(itertools.combinations(range(num_points), 2))

    num_edges_to_create = int(len(possible_edges) * density)

    indices = np.random.choice(len(possible_edges), size=num_edges_to_create, replace=False)
    selected_edges = [possible_edges[i] for i in indices]

    for i, j in selected_edges:
        distance = np.random.randint(1, max_distance + 1)
        dist_matrix[i, j] = distance
        dist_matrix[j, i] = distance

    return dist_matrix


def create_tsp_constraint_bqm(
    variables: dimod.variables.Variables, network_type: sparse_qubo.NetworkType
) -> dimod.BinaryQuadraticModel:
    num_points = int(np.sqrt(len(variables)))
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

    for row in range(num_points):
        row_vars = variables[row * num_points : (row + 1) * num_points]
        bqm += sparse_qubo.create_constraint_dwave(row_vars, sparse_qubo.ConstraintType.EQUAL_TO, network_type, c1=1)
    for col in range(num_points):
        col_vars = variables[col::num_points]
        bqm += sparse_qubo.create_constraint_dwave(col_vars, sparse_qubo.ConstraintType.EQUAL_TO, network_type, c1=1)

    return bqm


def create_tsp_cost_bqm(variables: dimod.variables.Variables, dist_matrix: np.ndarray) -> dimod.BinaryQuadraticModel:
    num_points = dist_matrix.shape[0]
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

    for t in range(num_points):
        next_t = (t + 1) % num_points
        for u in range(num_points):
            for v in range(num_points):
                if u == v:
                    continue

                dist = dist_matrix[u, v]
                if np.isinf(dist):
                    continue

                u_var = variables[u * num_points + t]
                v_var = variables[v * num_points + next_t]

                bqm.add_quadratic(u_var, v_var, dist)

    return bqm


def create_tsp_problem_bqm(
    network_type: sparse_qubo.NetworkType,
    dist_matrix: np.ndarray,
) -> dimod.BinaryQuadraticModel:
    num_points = dist_matrix.shape[0]

    variables = dimod.variables.Variables([f"X_{i}_{j}" for i in range(num_points) for j in range(num_points)])
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    bqm += create_tsp_constraint_bqm(variables, network_type)
    bqm += create_tsp_cost_bqm(variables, dist_matrix)

    return bqm
