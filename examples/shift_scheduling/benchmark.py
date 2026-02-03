from dataclasses import dataclass

import numpy as np
import pandas as pd

import sparse_qubo

from ..common import AnalysisDwaveResult, benchmark_sampling, plot_best_energy_histogram
from .problem import (
    calculate_solution_brute_force,
    check_feasibility,
    check_incompatible_pairs,
    create_incompatible_pairs,
    create_scheduling_cost_matrix,
    create_scheduling_problem_bqm,
)


@dataclass(frozen=True)
class ProblemParameters:
    num_days: int = 6
    num_workers: int = 2 * num_days
    num_incompatible_pairs: int = 5
    shifts_per_day: int = 2
    shifts_per_worker: int = 1
    naive_penalty_factor: float = 40  # You should adjust this value based on results
    dc_penalty_factor: float = 10  # You should adjust this value based on results
    seed: int = 42
    num_runs: int = 30
    solver_name: str = "Advantage2_system1.11"
    num_reads: int = 1000
    annealing_time: float = 20
    naive_chain_strength: float | None = None
    dc_chain_strength: float | None = None

    @property
    def cost_matrix(self) -> np.ndarray:
        return create_scheduling_cost_matrix(self.num_days, self.num_workers, seed=self.seed)

    @property
    def incompatible_pairs(self) -> set[tuple[int, int]]:
        return create_incompatible_pairs(self.num_workers, self.num_incompatible_pairs, seed=self.seed)


def run_benchmark(parameters: ProblemParameters):
    print("cost_matrix\n", parameters.cost_matrix)
    print("incompatible_pairs\n", parameters.incompatible_pairs)

    naive_bqm = create_scheduling_problem_bqm(
        sparse_qubo.NetworkType.NAIVE,
        parameters.num_days,
        parameters.num_workers,
        parameters.shifts_per_day,
        parameters.shifts_per_worker,
        parameters.incompatible_pairs,
        parameters.cost_matrix,
        penalty_factor=parameters.naive_penalty_factor,
    )
    dc_bqm = create_scheduling_problem_bqm(
        sparse_qubo.NetworkType.DIVIDE_AND_CONQUER,
        parameters.num_days,
        parameters.num_workers,
        parameters.shifts_per_day,
        parameters.shifts_per_worker,
        parameters.incompatible_pairs,
        parameters.cost_matrix,
        penalty_factor=parameters.dc_penalty_factor,
    )

    print("Starting benchmark...")
    print("Solver:", parameters.solver_name)
    print("Num Reads:", parameters.num_reads)
    print("Annealing Time:", parameters.annealing_time)
    print("Naive Chain Strength:", parameters.naive_chain_strength)
    print("D&C Chain Strength:", parameters.dc_chain_strength)

    print("Benchmarking Naive BQM...")
    benchmark_sampling(
        parameters.num_runs,
        "shift_scheduling/dwave/naive_bqm",
        parameters.solver_name,
        naive_bqm,
        parameters.num_reads,
        parameters.annealing_time,
        parameters.naive_chain_strength,
    )
    print("Benchmarking D&C BQM...")
    benchmark_sampling(
        parameters.num_runs,
        "shift_scheduling/dwave/dc_bqm",
        parameters.solver_name,
        dc_bqm,
        parameters.num_reads,
        parameters.annealing_time,
        parameters.dc_chain_strength,
    )

    print("Benchmarking completed.")


def analyze_results(parameters: ProblemParameters):
    def check_constraint(
        solution_row: pd.Series,
        num_days: int,
        num_workers: int,
        shifts_per_day: int,
        shifts_per_worker: int,
        incompatible_pairs: set[tuple[int, int]],
    ) -> bool:
        return check_feasibility(
            solution_row, num_days, num_workers, shifts_per_day, shifts_per_worker
        ) and check_incompatible_pairs(solution_row, num_days, num_workers, incompatible_pairs)

    naive_results = [
        AnalysisDwaveResult.from_sampleset(
            "shift_scheduling/dwave/naive_bqm",
            i,
            check_constraint,
            parameters.num_days,
            parameters.num_workers,
            parameters.shifts_per_day,
            parameters.shifts_per_worker,
            parameters.incompatible_pairs,
        )
        for i in range(parameters.num_runs)
    ]
    dc_results = [
        AnalysisDwaveResult.from_sampleset(
            "shift_scheduling/dwave/dc_bqm",
            i,
            check_constraint,
            parameters.num_days,
            parameters.num_workers,
            parameters.shifts_per_day,
            parameters.shifts_per_worker,
            parameters.incompatible_pairs,
        )
        for i in range(parameters.num_runs)
    ]

    naive_best_energy_list = [result.best_energy for result in naive_results if result.is_best_energy_feasible]
    dc_best_energy_list = [result.best_energy for result in dc_results if result.is_best_energy_feasible]

    print(f"Naive best feasible rate: {len(naive_best_energy_list) / parameters.num_runs}")
    print(f"D&C best feasible rate: {len(dc_best_energy_list) / parameters.num_runs}")

    plot_best_energy_histogram(
        [naive_best_energy_list, dc_best_energy_list],
        ["Naive", "Proposed Method"],
        "shift_scheduling/best_energy_histogram",
        invert_sign=True,
    )


if __name__ == "__main__":
    parameters = ProblemParameters()
    # run_benchmark(parameters)
    analyze_results(parameters)
    print(parameters.cost_matrix)
    print(parameters.incompatible_pairs)
    optimal_energy, optimal_assignment = calculate_solution_brute_force(
        parameters.num_days,
        parameters.num_workers,
        parameters.shifts_per_day,
        parameters.shifts_per_worker,
        parameters.cost_matrix,
        parameters.incompatible_pairs,
    )
    print(f"Optimal energy: {optimal_energy}")
    print(f"Optimal assignment: {optimal_assignment}")
