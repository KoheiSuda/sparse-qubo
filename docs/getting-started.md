# Getting Started

This guide will help you get started with `sparse-qubo` and understand its core concepts.

## Installation

Install `sparse-qubo` using pip:

```bash
pip install sparse-qubo
```

Or using `uv`:

```bash
uv add sparse-qubo
```

## Core Concepts

### QUBO Formulation

QUBO (Quadratic Unconstrained Binary Optimization) is a mathematical formulation used in quantum annealing and other optimization methods. A QUBO problem is defined as:

$$
\min \sum_{i} a_i x_i + \sum_{i<j} b_{ij} x_i x_j
$$

where $x_i \in \{0, 1\}$ are binary variables.

### Constraint Types

`sparse-qubo` supports several constraint types:

- **ONE_HOT**: Exactly one variable must be 1
- **EQUAL_TO**: Sum of variables equals a specific value
- **LESS_EQUAL**: Sum of variables is less than or equal to a value
- **GREATER_EQUAL**: Sum of variables is greater than or equal to a value
- **CLAMP**: Sum of variables is between two values (inclusive)

### Network Types

Different network architectures are available, each optimized for different scenarios:

- **DIVIDE_AND_CONQUER**: General-purpose, efficient for most cases
- **BUBBLE_SORT**: Simple sorting network
- **BITONIC_SORT**: Efficient for power-of-2 sizes
- **BENES**: Optimal for power-of-2 sizes
- **ODDEVEN_MERGE_SORT**: Batcher's odd-even merge sort
- **CLOS_NETWORK_MAX_DEGREE**: Optimized for maximum degree constraints
- **CLOS_NETWORK_MIN_EDGE**: Optimized for minimum edge count

## Basic Usage

### Creating Constraints

The main entry point is the `constraint` function from `sparse_qubo.dwave.constraint`:

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
```

### Working with D-Wave

The `constraint` function returns a `dimod.BinaryQuadraticModel` that can be used directly with D-Wave samplers:

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Create constraint
variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)

# Use with a sampler
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Get the best solution
best_sample = sampleset.first.sample
print(f"Best solution: {best_sample}")
```

### Using Different Constraint Types

#### One-Hot Constraint

Exactly one variable must be 1:

```python
bqm = constraint(variables, ConstraintType.ONE_HOT)
```

#### Equal-To Constraint

Sum of variables equals a specific value:

```python
# Sum equals 2
bqm = constraint(variables, ConstraintType.EQUAL_TO, c1=2)
```

#### Less-Equal Constraint

Sum of variables is less than or equal to a value:

```python
# Sum <= 3
bqm = constraint(variables, ConstraintType.LESS_EQUAL, c1=3)
```

#### Greater-Equal Constraint

Sum of variables is greater than or equal to a value:

```python
# Sum >= 1
bqm = constraint(variables, ConstraintType.GREATER_EQUAL, c1=1)
```

#### Clamp Constraint

Sum of variables is between two values:

```python
# 1 <= Sum <= 3
bqm = constraint(variables, ConstraintType.CLAMP, c1=1, c2=3)
```

### Choosing Network Types

Different network types have different characteristics:

- **DIVIDE_AND_CONQUER**: Good default choice, works for all constraint types
- **BUBBLE_SORT**: Simple but may produce more variables
- **BITONIC_SORT** / **BENES** / **ODDEVEN_MERGE_SORT**: Require power-of-2 sizes, but are efficient
- **CLOS_NETWORK_MAX_DEGREE**: Optimized when you need to limit maximum degree
- **CLOS_NETWORK_MIN_EDGE**: Optimized when you want to minimize edge count

Example:

```python
# For power-of-2 sizes, Benes network is optimal
variables = dimod.variables.Variables([f"x{i}" for i in range(8)])  # 8 is power of 2
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.BENES)
```

### Advanced Options

#### Threshold Parameter

For recursive networks, you can set a threshold to stop recursion early:

```python
bqm = constraint(
    variables,
    ConstraintType.ONE_HOT,
    NetworkType.DIVIDE_AND_CONQUER,
    threshold=4  # Stop recursion when size <= 4
)
```

#### Reverse Parameter

Some networks support a reverse parameter (used internally):

```python
# This is typically handled automatically
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)
```

## Low-Level API

For more control, you can use the low-level API:

```python
from sparse_qubo.core.constraint import get_constraint_qubo, ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Get QUBO directly
qubo = get_constraint_qubo(
    ["x0", "x1", "x2", "x3"],
    ConstraintType.ONE_HOT,
    NetworkType.DIVIDE_AND_CONQUER
)

# Access QUBO components
print(f"Variables: {qubo.variables}")
print(f"Linear terms: {qubo.linear}")
print(f"Quadratic terms: {qubo.quadratic}")
print(f"Constant: {qubo.constant}")
```

## Next Steps

- Check out [Examples](examples.md) for more detailed use cases
- Read the [API Reference](modules.md) for complete documentation
- See [CONTRIBUTING.md](https://github.com/KoheiSuda/sparse-qubo/blob/main/CONTRIBUTING.md) if you want to contribute
