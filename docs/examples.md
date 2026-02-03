# Examples

This page gives an overview of the **repository examples** in the `examples/` directory.

## Repository examples

The repository includes full problem setups that show how to combine constraints with objectives and compare network types.

### Shift scheduling (`examples/shift_scheduling/`)

- **Problem**: Assign workers to shifts per day with row/column sums and incompatible worker pairs.
- **Files**:
  - `problem.py`: `create_scheduling_problem_bqm(network_type, ...)` builds a BQM using `sparse_qubo.create_constraint_dwave` for row and column “exactly k selected” constraints, plus incompatible-pair penalties and a cost matrix.
  - `demo.ipynb`: Jupyter notebook that compares **NAIVE** vs **DIVIDE_AND_CONQUER** on D-Wave (variables, interactions, embedding qubits, chain strength, chain break rate, energy).
  - `benchmark.py`: Script to run multiple trials and save/analyze results.
- **Usage**: Run the notebook or benchmark from the project root (or add the root to `sys.path`). See the notebook for parameter and solver setup.

### TSP (`examples/tsp/`)

- **Problem**: Traveling salesman on a graph (one visit per city, one city per step).
- **Files**:
  - `problem.py`: `create_tsp_problem_bqm(network_type, dist_matrix)` builds a BQM with row/column one-hot constraints via `sparse_qubo.create_constraint_dwave` and distance-based quadratic objective.
  - `benchmark.py`: Benchmark script for TSP formulations.
- **Usage**: Import from `examples.tsp.problem` after adding the project root to `sys.path`, or run the benchmark script.

Both examples illustrate the typical pattern: build constraint BQMs per row/column with `sparse_qubo.create_constraint_dwave`, add objective terms, and optionally compare **NAIVE** vs **DIVIDE_AND_CONQUER** (or other network types).

---
