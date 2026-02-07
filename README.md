# sparse-qubo

[![Release](https://img.shields.io/github/v/release/KoheiSuda/sparse-qubo)](https://img.shields.io/github/v/release/KoheiSuda/sparse-qubo)
[![Build status](https://img.shields.io/github/actions/workflow/status/KoheiSuda/sparse-qubo/main.yml?branch=main)](https://github.com/KoheiSuda/sparse-qubo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/KoheiSuda/sparse-qubo/branch/main/graph/badge.svg)](https://codecov.io/gh/KoheiSuda/sparse-qubo)
[![Commit activity](https://img.shields.io/github/commit-activity/m/KoheiSuda/sparse-qubo)](https://img.shields.io/github/commit-activity/m/KoheiSuda/sparse-qubo)
[![License](https://img.shields.io/github/license/KoheiSuda/sparse-qubo)](https://img.shields.io/github/license/KoheiSuda/sparse-qubo)

**Sparse QUBO formulation for efficient embedding on hardware.**

`sparse-qubo` is a Python library that provides sparse QUBO (Quadratic Unconstrained Binary Optimization) formulations specifically for N-hot equality and inequality constraints. Constraint QUBOs are built from **switching networks**: each network is a list of **Switch** elements (left/right variable sets and constants), and the library converts them into QUBOs optimized for embedding on quantum annealing hardware (e.g. D-Wave) or other solvers.

The method is based on the paper [*Sparse QUBO Formulation for Efficient Embedding via Network-Based Decomposition of Equality and Inequality Constraints*](https://arxiv.org/abs/2601.18108) (Suda, Naito, Hasegawa, 2026).

## Features

- **Constraint types**: One-hot, equal-to, less-equal, greater-equal, clamp
- **Network types**: Naive (single linear equality), divide-and-conquer, bubble sort, bitonic sort, Benes, odd-even merge sort, Clos (max degree / minimum edge)
- **Backends**: D-Wave (`dimod.BQM`) and Fixstars Amplify (`amplify.Model`)
- **Examples**: Shift scheduling and TSP in `examples/` with notebooks and benchmarks

## Installation

```bash
pip install sparse-qubo
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add sparse-qubo
```

## Quick Start

```python
import dimod
import sparse_qubo

variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])

# One-hot constraint (divide-and-conquer network)
bqm = sparse_qubo.create_constraint_dwave(
    variables,
    sparse_qubo.ConstraintType.ONE_HOT,
    sparse_qubo.NetworkType.DIVIDE_AND_CONQUER,
)

# Equal-to: sum = 2
bqm = sparse_qubo.create_constraint_dwave(
    variables, sparse_qubo.ConstraintType.EQUAL_TO,
    sparse_qubo.NetworkType.DIVIDE_AND_CONQUER,
    c1=2,
)
```

For Fixstars Amplify, use `sparse_qubo.create_constraint_amplify` with a list of `amplify.Variable`. See the [documentation](https://KoheiSuda.github.io/sparse-qubo/) for more examples and the full API.

## Examples

The `examples/` directory includes:

- **Shift scheduling** (`examples/shift_scheduling/`): Demo notebook comparing NAIVE vs DIVIDE_AND_CONQUER on D-Wave, plus `create_scheduling_problem_bqm` and benchmarks
- **TSP** (`examples/tsp/`): Problem builder and benchmarks for traveling salesman formulations


## Documentation

- **Documentation**: <https://KoheiSuda.github.io/sparse-qubo/>
- **API reference**: [modules](https://KoheiSuda.github.io/sparse-qubo/modules/)

## Reference

**Kohei Suda, Soshun Naito, Yoshihiko Hasegawa.** *Sparse QUBO Formulation for Efficient Embedding via Network-Based Decomposition of Equality and Inequality Constraints.* arXiv:2601.18108, 2026. <https://arxiv.org/abs/2601.18108>

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/KoheiSuda/sparse-qubo/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms in [LICENSE](https://github.com/KoheiSuda/sparse-qubo/blob/main/LICENSE).

## Links

- **GitHub**: <https://github.com/KoheiSuda/sparse-qubo>
- **PyPI**: <https://pypi.org/project/sparse-qubo/>
