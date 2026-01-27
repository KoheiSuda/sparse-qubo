# sparse-qubo

[![Release](https://img.shields.io/github/v/release/KoheiSuda/sparse-qubo)](https://img.shields.io/github/v/release/KoheiSuda/sparse-qubo)
[![Build status](https://img.shields.io/github/actions/workflow/status/KoheiSuda/sparse-qubo/main.yml?branch=main)](https://github.com/KoheiSuda/sparse-qubo/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/KoheiSuda/sparse-qubo/branch/main/graph/badge.svg)](https://codecov.io/gh/KoheiSuda/sparse-qubo)
[![License](https://img.shields.io/github/license/KoheiSuda/sparse-qubo)](https://img.shields.io/github/license/KoheiSuda/sparse-qubo)

**Sparse QUBO formulation for efficient embedding on hardware.**

`sparse-qubo` is a Python library that provides sparse QUBO (Quadratic Unconstrained Binary Optimization) formulations for various constraint types, optimized for embedding on quantum annealing hardware such as D-Wave systems.

## Features

- **Multiple Constraint Types**: Support for one-hot, equal-to, less-equal, greater-equal, and clamp constraints
- **Various Network Architectures**: Multiple switching network implementations including:
  - Benes network
  - Bitonic sort network
  - Bubble sort network
  - Clos network (max degree and minimum edge variants)
  - Divide-and-conquer network
  - Odd-even merge sort network
- **Hardware Optimization**: Designed to minimize the number of variables and edges for efficient hardware embedding
- **D-Wave Integration**: Direct integration with D-Wave's `dimod` library

## Installation

```bash
pip install sparse-qubo
```

Or using `uv`:

```bash
uv add sparse-qubo
```

## Quick Start

### Basic Usage

Create a one-hot constraint using the divide-and-conquer network:

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Define variables
variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])

# Create a one-hot constraint
bqm = constraint(
    variables,
    ConstraintType.ONE_HOT,
    NetworkType.DIVIDE_AND_CONQUER
)

# Use with D-Wave sampler
# sampler = dimod.SimulatedAnnealingSampler()
# sampleset = sampler.sample(bqm)
```

### Different Constraint Types

```python
# Equal-to constraint: sum of variables equals 2
bqm = constraint(variables, ConstraintType.EQUAL_TO, c1=2)

# Less-equal constraint: sum of variables <= 3
bqm = constraint(variables, ConstraintType.LESS_EQUAL, c1=3)

# Greater-equal constraint: sum of variables >= 1
bqm = constraint(variables, ConstraintType.GREATER_EQUAL, c1=1)

# Clamp constraint: 1 <= sum of variables <= 3
bqm = constraint(variables, ConstraintType.CLAMP, c1=1, c2=3)
```

### Different Network Types

```python
# Use Benes network (requires power of 2 size)
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.BENES)

# Use bubble sort network
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.BUBBLE_SORT)

# Use Clos network with maximum degree constraint
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.CLOS_NETWORK_MAX_DEGREE)
```

## Documentation

- [Getting Started](getting-started.md) - Detailed guide on using the library
- [Examples](examples.md) - Code examples and use cases
- [API Reference](modules.md) - Complete API documentation

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/KoheiSuda/sparse-qubo/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms specified in the LICENSE file.

## Links

- **GitHub Repository**: <https://github.com/KoheiSuda/sparse-qubo/>
- **Documentation**: <https://KoheiSuda.github.io/sparse-qubo/>
- **PyPI**: <https://pypi.org/project/sparse-qubo/>
