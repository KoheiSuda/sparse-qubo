import itertools
from itertools import combinations

import dimod
import dimod.variables
import numpy as np
import pandas as pd

import sparse_qubo


def create_scheduling_cost_matrix(
    num_row: int, num_col: int, max_cost: int = -1, min_cost: int = -5, seed: int | None = None
) -> np.ndarray:
    assert num_row >= 2
    assert num_col >= 2
    assert max_cost >= min_cost

    np.random.seed(seed)
    random_matrix = np.random.randint(min_cost, max_cost + 1, (num_row, num_col))

    return random_matrix.astype(float)


def check_feasibility(solution_row: pd.Series, num_row: int, num_col: int, c_row: int, c_col: int) -> bool:
    for i in range(num_row):
        row_sum = sum(solution_row[f"X_{i}_{j}"] for j in range(num_col))
        if row_sum != c_row:
            return False
    for j in range(num_col):
        col_sum = sum(solution_row[f"X_{i}_{j}"] for i in range(num_row))
        if col_sum != c_col:
            return False
    return True


def create_incompatible_pairs(num_workers: int, num_combinations: int, seed: int | None = None) -> set[tuple[int, int]]:
    assert num_workers >= 2
    assert 1 <= num_combinations <= num_workers * (num_workers - 1) // 2

    np.random.seed(seed)
    all_pairs = list(itertools.combinations(range(num_workers), 2))
    np.random.shuffle(all_pairs)

    selected_pairs = set()
    usage_counts = np.zeros(num_workers, dtype=int)

    for _ in range(num_combinations):
        best_pair_idx = -1
        min_score = (float("inf"), float("inf"))
        for idx, (p1, p2) in enumerate(all_pairs):
            c1 = usage_counts[p1]
            c2 = usage_counts[p2]
            score = (max(c1, c2), c1 + c2)
            if score < min_score:
                min_score = score
                best_pair_idx = idx
            if score == (0, 0):
                break

        p1, p2 = all_pairs.pop(best_pair_idx)
        selected_pairs.add((p1, p2))
        usage_counts[p1] += 1
        usage_counts[p2] += 1

    return selected_pairs


def check_incompatible_pairs(
    solution_row: pd.Series, num_row: int, num_col: int, incompatible_pairs: set[tuple[int, int]]
) -> bool:
    for i in range(num_row):
        workers_on_day = []
        for j in range(num_col):
            if solution_row[f"X_{i}_{j}"] == 1:
                workers_on_day.append(j)
        for p1, p2 in itertools.combinations(workers_on_day, 2):
            if (p1, p2) in incompatible_pairs or (p2, p1) in incompatible_pairs:
                return False
    return True


def create_scheduling_problem_bqm(
    network_type: sparse_qubo.NetworkType,
    num_row: int,
    num_col: int,
    c_row: int,
    c_col: int,
    incompatible_pairs: set[tuple[int, int]],
    cost_matrix: np.ndarray,
    penalty_factor: float = 1,
    seed: int = 42,
) -> dimod.BinaryQuadraticModel:
    assert num_row >= 2
    assert num_col >= 2
    assert c_row >= 1
    assert c_col >= 1

    variables = dimod.variables.Variables([f"X_{i}_{j}" for i in range(num_row) for j in range(num_col)])
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    for row in range(num_row):
        row_vars = variables[row * num_col : (row + 1) * num_col]
        bqm += sparse_qubo.create_constraint_dwave(
            row_vars, sparse_qubo.ConstraintType.EQUAL_TO, network_type, c1=c_row
        )
    for col in range(num_col):
        col_vars = variables[col::num_col]
        bqm += sparse_qubo.create_constraint_dwave(
            col_vars, sparse_qubo.ConstraintType.EQUAL_TO, network_type, c1=c_col
        )

    for p1, p2 in incompatible_pairs:
        for row in range(num_row):
            bqm.add_quadratic(variables[row * num_col + p1], variables[row * num_col + p2], bias=1)

    bqm *= penalty_factor

    for i in range(num_row):
        for j in range(num_col):
            bqm.add_linear(variables[i * num_col + j], cost_matrix[i, j])

    return bqm


def calculate_solution_brute_force(
    num_row: int,
    num_col: int,
    c_row: int,
    c_col: int,
    cost_matrix: np.ndarray,
    incompatible_pairs: set[tuple[int, int]],
):
    assert num_col * c_col == num_row * c_row, "総枠数が一致しません"
    assert cost_matrix.shape == (num_row, num_col), "行列のサイズが不正です"

    initial_shifts = [c_col] * num_col
    min_cost = [float("inf")]
    best_assignment = [None]

    min_val = np.min(cost_matrix)

    def solve(day, current_shifts, current_total_cost, path):
        remaining_days = num_row - day
        if current_total_cost + (remaining_days * min_val * c_row) >= min_cost[0]:
            return

        if day == num_row:
            if current_total_cost < min_cost[0]:
                min_cost[0] = current_total_cost
                best_assignment[0] = [list(d) for d in path]
            return

        available_emps = [i for i, s in enumerate(current_shifts) if s > 0]

        for today_workers in combinations(available_emps, c_row):
            day_cost = 0
            for emp_id in today_workers:
                day_cost += cost_matrix[day][emp_id]

            for p1, p2 in combinations(today_workers, 2):
                if (p1, p2) in incompatible_pairs or (p2, p1) in incompatible_pairs:
                    day_cost += 100.0

            for emp_id in today_workers:
                current_shifts[emp_id] -= 1
            path[day] = today_workers

            solve(day + 1, current_shifts, current_total_cost + day_cost, path)

            for emp_id in today_workers:
                current_shifts[emp_id] += 1

    solve(0, initial_shifts, 0.0, [None] * num_row)

    return min_cost[0], best_assignment[0]
