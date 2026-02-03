from pathlib import Path

import dimod
import dimod.variables
import dwave.system
import matplotlib.pyplot as plt
import pandas as pd

import sparse_qubo

from ..common import run_embedding
from .problem import create_tsp_constraint_bqm, create_tsp_cost_bqm, create_tsp_problem, create_tsp_problem_bqm


def analyze_naive_qubo_quadratic_terms(
    num_points_list: list[int], density: float, max_distance: int, seed: int | None = None
) -> None:
    num_constraint_quadratic_terms: list[dict[str, int]] = []
    num_cost_quadratic_terms: list[dict[str, int]] = []
    num_total_quadratic_terms: list[dict[str, int]] = []

    for num_points in num_points_list:
        dist_matrix = create_tsp_problem(num_points, density, max_distance, seed)
        variables = dimod.variables.Variables([f"X_{i}_{j}" for i in range(num_points) for j in range(num_points)])
        bqm_constraint = create_tsp_constraint_bqm(variables, sparse_qubo.NetworkType.NAIVE)
        bqm_cost = create_tsp_cost_bqm(variables, dist_matrix)
        total_bqm = bqm_constraint + bqm_cost
        num_constraint_quadratic_terms.append(bqm_constraint.num_interactions)
        num_cost_quadratic_terms.append(bqm_cost.num_interactions)
        num_total_quadratic_terms.append(total_bqm.num_interactions)

    output_dir = "outputs"
    output_filepath = Path(f"{output_dir}/tsp/quadratic_terms.png")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))

    plt.plot(num_points_list, num_constraint_quadratic_terms, marker="o", label="Constraint Terms")
    plt.plot(num_points_list, num_cost_quadratic_terms, marker="s", label="Cost Terms")
    plt.plot(num_points_list, num_total_quadratic_terms, marker="^", label="Total Terms")

    plt.xlabel("N")
    plt.xticks(num_points_list)
    plt.ylabel("Number of Quadratic Terms")
    # plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()


def run_embedding_benchmark(
    num_points_list: list[int], density: float, max_distance: int, solver_name: str, seed: int | None = None
) -> None:
    results_list: list[int] = []
    num_trials: int = 5

    sampler = dwave.system.DWaveSampler(solver=solver_name)
    graph = sampler.to_networkx_graph()

    flag = False
    for num_points in num_points_list:
        dist_matrix = create_tsp_problem(num_points, density, max_distance, seed)
        bqm = create_tsp_problem_bqm(sparse_qubo.NetworkType.NAIVE, dist_matrix)
        row = {}
        row["num_points"] = num_points
        for trial in range(num_trials):
            result = run_embedding(bqm, graph)
            if result is None:
                flag = True
                break
            row[f"trial_{trial}"] = result
        results_list.append(row)
        print(f"num_points: {num_points}")
        print(f"row: {row}")
        if flag:
            break


def plot_embedding_benchmark(num_points_list: list[int], df: pd.DataFrame) -> None:
    output_dir = "outputs"
    output_filepath = Path(f"{output_dir}/tsp/embedding_benchmark.png")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    trial_cols = [f"trial_{i}" for i in range(5)]
    df = df.set_index("num_points")
    df = df[trial_cols].div(num_points_list**2, axis=0)
    df["mean"] = df[trial_cols].mean(axis=1)
    df["std"] = df[trial_cols].std(axis=1)
    mean_aligned = df["mean"].reindex(num_points_list)
    std_aligned = df["std"].reindex(num_points_list)

    plt.figure(figsize=(6, 4))

    plt.plot(num_points_list, mean_aligned.values, marker="o", color="red")
    plt.fill_between(
        num_points_list,
        (mean_aligned - std_aligned).values,
        (mean_aligned + std_aligned).values,
        alpha=0.2,
        color="red",
    )

    plt.xlabel("N")
    plt.xticks(num_points_list)
    plt.ylabel("Number of Qubits")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()


if __name__ == "__main__":
    num_points_list = list(range(4, 22, 2))
    density = 0.3
    max_distance = 100
    seed = 42
    solver_name = "Advantage2_system1.11"
    analyze_naive_qubo_quadratic_terms(
        num_points_list=num_points_list, density=density, max_distance=max_distance, seed=seed
    )
    # run_embedding_benchmark(
    #     num_points_list=num_points_list, density=density, max_distance=max_distance, solver_name=solver_name, seed=seed
    # )
    df = pd.read_csv("outputs/tsp/embedding_results.csv")
    plot_embedding_benchmark(num_points_list, df)
