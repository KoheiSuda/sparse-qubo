import math
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import dimod
import dwave
import dwave.system
import matplotlib.pyplot as plt
import minorminer
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def sampling_with_dwave(
    solver_name: str,
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 1000,
    annealing_time: float = 20,
    chain_strength: float | None = None,
) -> dimod.SampleSet:
    with dwave.system.EmbeddingComposite(dwave.system.DWaveSampler(solver=solver_name)) as sampler:
        result: dimod.SampleSet = sampler.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=annealing_time,
            chain_strength=chain_strength,
            return_embedding=True,
        )

    return result


def calculate_chain_break_rate(df: pd.DataFrame) -> float:
    total_weighted_breaks: float = (df["chain_break_fraction"] * df["num_occurrences"]).sum()
    total_samples: float = df["num_occurrences"].sum()

    return total_weighted_breaks / total_samples


def benchmark_sampling(
    num_runs: int,
    filename: str,
    solver_name: str,
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int = 1000,
    annealing_time: float = 20,
    chain_strength: float | None = None,
) -> None:
    output_dir = "outputs"
    for run in range(num_runs):
        print(f"\nRunning trial {run + 1} of {num_runs}")

        result = sampling_with_dwave(solver_name, bqm, num_reads, annealing_time, chain_strength)

        embedding = result.info["embedding_context"]["embedding"]
        print("number of qubits after embedding:", sum([len(v) for v in embedding.values()]))

        df = result.to_pandas_dataframe()
        print("energy average:", df["energy"].mean())

        output_filepath = Path(f"{output_dir}/{filename}_{run}.pkl")
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "wb") as f:
            pickle.dump(result.to_serializable(), f)


@dataclass
class AnalysisDwaveResult:
    feasible_rate: float
    embedding_qubits: int
    average_chain_length: float
    max_chain_length: int
    std_chain_length: float
    chain_break_rate: float
    chain_strength: float
    best_energy: float
    average_energy: float
    std_energy: float
    is_best_energy_feasible: bool

    @classmethod
    def from_sampleset(
        cls,
        file_name: str,
        run_idx: int,
        check_constraint: Callable[[pd.Series, ...], bool],
        *args: Any,
    ) -> Self:
        output_dir = "outputs"
        with open(f"{output_dir}/{file_name}_{run_idx}.pkl", "rb") as f:
            sampleset: dimod.SampleSet = dimod.SampleSet.from_serializable(pickle.load(f))  # noqa: S301

        embedding = sampleset.info["embedding_context"]["embedding"]
        embedding_qubits = sum([len(v) for v in embedding.values()])

        chain_strength = sampleset.info["embedding_context"]["chain_strength"]

        chain_lengths = [len(chain) for chain in embedding.values()]
        average_chain_length = np.mean(chain_lengths)
        max_chain_length = np.max(chain_lengths)
        std_chain_length = np.std(chain_lengths)

        df = sampleset.to_pandas_dataframe()

        total_weighted_breaks = (df["chain_break_fraction"] * df["num_occurrences"]).sum()
        total_samples = df["num_occurrences"].sum()
        chain_break_rate = total_weighted_breaks / total_samples

        num_feasible_samples = 0
        for _, solution_row in df.iterrows():
            if check_constraint(solution_row, *args):
                num_feasible_samples += solution_row["num_occurrences"]
        feasible_rate = num_feasible_samples / total_samples

        return cls(
            feasible_rate=feasible_rate,
            embedding_qubits=embedding_qubits,
            average_chain_length=float(average_chain_length),
            max_chain_length=int(max_chain_length),
            std_chain_length=float(std_chain_length),
            chain_break_rate=chain_break_rate,
            chain_strength=chain_strength,
            best_energy=float(sampleset.first.energy),
            average_energy=float(df["energy"].mean()),
            std_energy=float(df["energy"].std()),
            is_best_energy_feasible=check_constraint(sampleset.first.sample, *args),
        )


def plot_best_energy_histogram(
    results_list: list[list[float]],
    labels: list[str],
    filename: str,
    invert_sign: bool = False,  # if True, the sign of the energy is inverted
) -> None:
    if invert_sign:
        results_list = [[-val for val in data] for data in results_list]

    plt.tight_layout()
    plt.show()

    output_dir = "outputs"
    output_filepath = Path(f"{output_dir}/{filename}.png")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    if len(results_list) != len(labels):
        raise ValueError(f"Length of results_list ({len(results_list)}) and labels ({len(labels)}) do not match.")

    all_data = [val for data in results_list for val in data]

    data_min_int = math.floor(min(all_data))
    data_max_int = math.ceil(max(all_data))

    integer_values = list(range(data_min_int, data_max_int + 1))

    bin_edges = [val - 0.5 for val in integer_values] + [data_max_int + 0.5]

    _, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.hist(
        results_list,
        bins=bin_edges,
        label=labels,
        alpha=0.9,
        linewidth=0.5,
        align="mid",
    )

    ax.set_xticks(integer_values)
    ax.set_xticklabels([str(val) for val in integer_values], fontsize=10)
    ax.set_xlabel("Value", fontsize=14)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Frequency", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    ax.legend(
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        fontsize=12,
    )

    ax.tick_params(direction="in", top=True, right=True, which="both")

    plt.savefig(output_filepath)


def run_embedding(bqm: dimod.BinaryQuadraticModel, graph: nx.Graph, seed: int | None = None) -> int | None:
    try:
        embedding, is_success = minorminer.find_embedding(
            dimod.to_networkx_graph(bqm), graph, return_overlap=True, random_seed=seed
        )
        if not is_success:
            print("Embedding failed")
            return None
    except Exception as e:
        print(e)
        return None

    return sum([len(v) for v in embedding.values()])
