# Examples

This page provides practical examples of using `sparse-qubo` for various scenarios.

## Example 1: One-Hot Constraint

A one-hot constraint ensures exactly one variable is set to 1. This is useful for selection problems.

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Select exactly one item from 4 options
variables = dimod.variables.Variables(["select_item_0", "select_item_1", "select_item_2", "select_item_3"])

# Create one-hot constraint
bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)

# Solve
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Verify: exactly one variable is 1
for sample in sampleset:
    assert sum(sample.sample.values()) == 1
```

## Example 2: Equal-To Constraint

An equal-to constraint ensures the sum equals a specific value.

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Select exactly 2 items from 5 options
variables = dimod.variables.Variables([f"select_{i}" for i in range(5)])

# Create equal-to constraint: sum = 2
bqm = constraint(variables, ConstraintType.EQUAL_TO, NetworkType.DIVIDE_AND_CONQUER, c1=2)

# Solve
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Verify: exactly 2 variables are 1
for sample in sampleset:
    assert sum(sample.sample.values()) == 2
```

## Example 3: Less-Equal Constraint

A less-equal constraint ensures the sum is at most a value.

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Select at most 3 items from 6 options
variables = dimod.variables.Variables([f"select_{i}" for i in range(6)])

# Create less-equal constraint: sum <= 3
bqm = constraint(variables, ConstraintType.LESS_EQUAL, NetworkType.DIVIDE_AND_CONQUER, c1=3)

# Solve
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Verify: at most 3 variables are 1
for sample in sampleset:
    assert sum(sample.sample.values()) <= 3
```

## Example 4: Clamp Constraint

A clamp constraint ensures the sum is between two values.

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Select between 2 and 4 items from 7 options
variables = dimod.variables.Variables([f"select_{i}" for i in range(7)])

# Create clamp constraint: 2 <= sum <= 4
bqm = constraint(variables, ConstraintType.CLAMP, NetworkType.DIVIDE_AND_CONQUER, c1=2, c2=4)

# Solve
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)

# Verify: sum is between 2 and 4
for sample in sampleset:
    total = sum(sample.sample.values())
    assert 2 <= total <= 4
```

## Example 5: Comparing Network Types

Different network types produce different QUBO formulations. Compare their sizes:

```python
from sparse_qubo.core.constraint import get_constraint_qubo, ConstraintType
from sparse_qubo.core.base_network import NetworkType

variables = [f"x{i}" for i in range(8)]  # Power of 2

networks = [
    NetworkType.DIVIDE_AND_CONQUER,
    NetworkType.BUBBLE_SORT,
    NetworkType.BENES,
    NetworkType.BITONIC_SORT,
]

for network_type in networks:
    qubo = get_constraint_qubo(variables, ConstraintType.ONE_HOT, network_type)
    num_vars = len(qubo.variables)
    num_edges = len(qubo.quadratic)
    print(f"{network_type.value:25s}: {num_vars:3d} variables, {num_edges:3d} edges")
```

## Example 6: Using with Custom Objective

Combine constraints with custom objective functions:

```python
import dimod
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Variables for selecting items
variables = dimod.variables.Variables(["item_0", "item_1", "item_2", "item_3"])

# Create one-hot constraint
constraint_bqm = constraint(variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER)

# Create custom objective: minimize cost
# Assume costs: [10, 5, 8, 12]
costs = [10, 5, 8, 12]
objective_bqm = dimod.BinaryQuadraticModel(
    {f"item_{i}": costs[i] for i in range(4)},
    {},
    0.0,
    dimod.BINARY
)

# Combine: objective + lambda * constraint
lambda_multiplier = 100  # Penalty weight
combined_bqm = objective_bqm + lambda_multiplier * constraint_bqm

# Solve
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(combined_bqm, num_reads=100)

# Find minimum cost solution
best = sampleset.first
print(f"Best solution: {best.sample}")
print(f"Cost: {sum(costs[i] * best.sample[f'item_{i}'] for i in range(4))}")
```

## Example 7: Power-of-2 Optimization

For power-of-2 sizes, use specialized networks:

```python
from sparse_qubo.core.constraint import get_constraint_qubo, ConstraintType
from sparse_qubo.core.base_network import NetworkType

# Use Benes network for power-of-2 size (optimal)
variables = [f"x{i}" for i in range(16)]  # 16 = 2^4

qubo = get_constraint_qubo(variables, ConstraintType.ONE_HOT, NetworkType.BENES)

print(f"Variables: {len(qubo.variables)}")
print(f"Quadratic terms: {len(qubo.quadratic)}")
```

## Example 8: Threshold Parameter

Use threshold to control recursion depth:

```python
from sparse_qubo.core.constraint import get_constraint_qubo, ConstraintType
from sparse_qubo.core.base_network import NetworkType

variables = [f"x{i}" for i in range(16)]

# Without threshold (full recursion)
qubo1 = get_constraint_qubo(
    variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER
)

# With threshold (stops recursion at size 4)
qubo2 = get_constraint_qubo(
    variables, ConstraintType.ONE_HOT, NetworkType.DIVIDE_AND_CONQUER, threshold=4
)

print(f"Without threshold: {len(qubo1.variables)} variables")
print(f"With threshold=4: {len(qubo2.variables)} variables")
```

## Example 9: Clos Network Configuration

Configure Clos network with maximum degree:

```python
from sparse_qubo.networks.clos_network_max_degree import ClosNetworkWithMaxDegree
from sparse_qubo.core.node import VariableNode, NodeAttribute
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType
from sparse_qubo.core.constraint import get_initial_nodes

# Set maximum degree
ClosNetworkWithMaxDegree.reset_max_degree(5)

# Create variables
variables = [f"x{i}" for i in range(10)]
left_nodes, right_nodes = get_initial_nodes(variables, ConstraintType.ONE_HOT)

# Generate network
channels = ClosNetworkWithMaxDegree.generate_network(left_nodes, right_nodes)
print(f"Number of channels: {len(channels)}")
```

## Example 10: Error Handling

Handle invalid inputs gracefully:

```python
from sparse_qubo.dwave.constraint import constraint
from sparse_qubo.core.constraint import ConstraintType
from sparse_qubo.core.base_network import NetworkType
import dimod

variables = dimod.variables.Variables(["x0", "x1", "x2"])

# Invalid: c1 > number of variables
try:
    bqm = constraint(variables, ConstraintType.EQUAL_TO, NetworkType.DIVIDE_AND_CONQUER, c1=5)
except ValueError as e:
    print(f"Error: {e}")

# Valid: c1 <= number of variables
bqm = constraint(variables, ConstraintType.EQUAL_TO, NetworkType.DIVIDE_AND_CONQUER, c1=2)
```
