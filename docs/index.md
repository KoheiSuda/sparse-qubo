# sparse-qubo

[![Release](https://img.shields.io/github/v/release/KoheiSuda/sparse-qubo)](https://img.shields.io/github/v/release/KoheiSuda/sparse-qubo)
[![Build status](https://img.shields.io/github/actions/workflow/status/KoheiSuda/sparse-qubo/main.yml?branch=main)](https://github.com/KoheiSuda/sparse-qubo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/KoheiSuda/sparse-qubo/branch/main/graph/badge.svg)](https://codecov.io/gh/KoheiSuda/sparse-qubo)
[![License](https://img.shields.io/github/license/KoheiSuda/sparse-qubo)](https://img.shields.io/github/license/KoheiSuda/sparse-qubo)

**Sparse QUBO formulation for efficient embedding on hardware.**

`sparse-qubo` is a Python library that provides sparse QUBO (Quadratic Unconstrained Binary Optimization) formulations specifically for N-hot equality and inequality constraints. Constraint QUBOs are built from **switching networks**: each network is a list of **Switch** elements (left/right variable sets and constants), and the library converts them into QUBOs optimized for embedding on quantum annealing hardware (e.g. D-Wave) or other solvers.

## Features

- **Multiple constraint types**: One-hot, equal-to, less-equal, greater-equal, and clamp
- **Switching networks**: Constraint formulations are implemented as switching networks (each network is a list of **Switch** objects), which yield small quadratic terms
- **Multiple network architectures**:
  - Benes network
  - Bitonic sort network
  - Bubble sort network
  - Clos network (max degree and minimum edge variants)
  - Divide-and-conquer network
  - Odd-even merge sort network
- **Backends**: D-Wave (`dimod.BQM`) and Fixstars Amplify (`amplify.Model`)
- **Examples**: Repository includes example problems (shift scheduling, TSP) with notebooks and benchmarks

## Installation

```bash
pip install sparse-qubo
```

Or using `uv`:

```bash
uv add sparse-qubo
```

## Quick Start

### D-Wave (dimod)

Create a one-hot constraint using the divide-and-conquer network:

```python
import dimod
import sparse_qubo

variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])

# One-hot constraint
bqm = sparse_qubo.create_constraint_dwave(
    variables,
    sparse_qubo.ConstraintType.ONE_HOT,
    sparse_qubo.NetworkType.DIVIDE_AND_CONQUER,
)
```

### Constraint types and network types

```python
# Equal-to: sum of variables equals 2
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.EQUAL_TO, sparse_qubo.NetworkType.DIVIDE_AND_CONQUER, c1=2)

# Less-equal: sum <= 3
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.LESS_EQUAL, sparse_qubo.NetworkType.DIVIDE_AND_CONQUER, c1=3)

# Naive formulation (single switch; no additional variables, denser quadratic terms)
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.ONE_HOT, sparse_qubo.NetworkType.NAIVE)

# Other networks (e.g. bubble sort network)
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.ONE_HOT, sparse_qubo.NetworkType.BUBBLE_SORT)
```

### Repository examples

The `examples/` directory contains full problem setups:

- **Shift scheduling** (`examples/shift_scheduling/`): Demo notebook, problem builder (`create_scheduling_problem_bqm` using `sparse_qubo.create_constraint_dwave`), and benchmarks comparing `NetworkType.NAIVE` vs `NetworkType.DIVIDE_AND_CONQUER` on D-Wave
- **TSP** (`examples/tsp/`): Problem builder and benchmarks for traveling salesman formulations

See [Examples](examples.md) for details and inline code samples.

## Reference

This library implements the sparse QUBO formulation described in:

**Kohei Suda, Soshun Naito, Yoshihiko Hasegawa.** *Sparse QUBO Formulation for Efficient Embedding via Network-Based Decomposition of Equality and Inequality Constraints.* arXiv:2601.18108, 2026. <https://arxiv.org/abs/2601.18108>

The paper provides a comprehensive description of the network-based constraint decomposition and the divide-and-conquer algorithm utilized in this library. Furthermore, it discusses the effectiveness of the method through experiments performed on D-Wave hardware.

## Documentation

- [Getting Started](getting-started.md) — Concepts, constraint and network types, low-level API
- [Examples](examples.md) — Inline examples and repository example overview
- [Usage](usage.md) — Constraint prefix counter and variable naming
- [API Reference](modules.md) — Module and class reference

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/KoheiSuda/sparse-qubo/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms specified in the LICENSE file.

## Links

- **GitHub**: <https://github.com/KoheiSuda/sparse-qubo/>
- **Documentation**: <https://KoheiSuda.github.io/sparse-qubo/>
- **PyPI**: <https://pypi.org/project/sparse-qubo/>
