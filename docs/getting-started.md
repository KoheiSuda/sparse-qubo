# Getting Started

This guide introduces core concepts and basic usage of `sparse-qubo`.

## Installation

Install with pip:

```bash
pip install sparse-qubo
```

Or with `uv`:

```bash
uv add sparse-qubo
```

## Core Concepts

### QUBO formulation

QUBO (Quadratic Unconstrained Binary Optimization) is a formulation used in quantum annealing and related methods. A QUBO is:

$$
\min \sum_{i} a_i x_i + \sum_{i<j} b_{ij} x_i x_j
$$

with binary variables $x_i \in \{0, 1\}$.

### Switching networks and Switch

Constraint QUBOs in this library are built from **switching networks**. A switching network is a list of **Switch** objects. Each Switch has:

- **Left and right variable sets**: Two disjoint sets of binary variable names and optional integer constants. The constraint is encoded by requiring that the sum on the left equals the sum on the right (up to constants) for each Switch.
- **QUBO conversion**: The function `switches_to_qubo(switches)` (from `sparse_qubo.core.switch`) turns a list of Switch elements into a single QUBO (variables, linear, quadratic, constant).

Different **network types** (e.g. divide-and-conquer, Benes, bubble sort) produce different sequences of Switch elements and thus different variable counts and sparsity. The **NAIVE** network type does not use switching networks; it encodes the constraint as a single squared term in the usual way (no additional variables, denser quadratic terms).

### Constraint types

Supported constraint types:

| Type | Description |
|------|-------------|
| **ONE_HOT** | Exactly one variable is 1 |
| **EQUAL_TO** | Sum of variables equals a value (`c1`) |
| **LESS_EQUAL** | Sum of variables ≤ value (`c1`) |
| **GREATER_EQUAL** | Sum of variables ≥ value (`c1`) |
| **CLAMP** | Sum of variables in [`c1`, `c2`] |

### Network types

| Network | Notes |
|---------|--------|
| **DIVIDE_AND_CONQUER** | General-purpose; good default |
| **BUBBLE_SORT** | Simple sorting network |
| **BITONIC_SORT** / **BENES** / **ODDEVEN_MERGE_SORT** | Require power-of-2 variable size and automatically add auxiliary variables|
| **CLOS_NETWORK_MAX_DEGREE** | Tune max degree (see network module) |
| **CLOS_NETWORK_MIN_EDGE** | Minimize edge count |
| **NAIVE** | No switching network; single squared term |

## Basic usage

### Creating constraints (D-Wave / dimod)

Use `sparse_qubo.create_constraint_dwave` to get a `dimod.BinaryQuadraticModel`:

```python
import dimod
import sparse_qubo

variables = dimod.variables.Variables(["x0", "x1", "x2", "x3"])

# One-hot
bqm = sparse_qubo.create_constraint_dwave(
    variables,
    sparse_qubo.ConstraintType.ONE_HOT,
    sparse_qubo.NetworkType.DIVIDE_AND_CONQUER,
)
```

### Constraint types (D-Wave)

```python
# Equal-to: sum = 2
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.EQUAL_TO, sparse_qubo.NetworkType.DIVIDE_AND_CONQUER, c1=2)

# Less-equal: sum <= 3
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.LESS_EQUAL, sparse_qubo.NetworkType.DIVIDE_AND_CONQUER, c1=3)

# Greater-equal: sum >= 1
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.GREATER_EQUAL, sparse_qubo.NetworkType.DIVIDE_AND_CONQUER, c1=1)

# Clamp: 1 <= sum <= 3
bqm = sparse_qubo.create_constraint_dwave(variables, sparse_qubo.ConstraintType.CLAMP, sparse_qubo.NetworkType.DIVIDE_AND_CONQUER, c1=1, c2=3)
```

### Choosing a network type

- **DIVIDE_AND_CONQUER**: Default for most use cases; works for any size and constraint type.
- **BUBBLE_SORT**: Simple; may use more variables.
- **BITONIC_SORT**, **BENES**, **ODDEVEN_MERGE_SORT**: Only for power-of-2 variable count; can reduce edges.
- **CLOS_NETWORK_MAX_DEGREE** / **CLOS_NETWORK_MIN_EDGE**: When you need to tune degree or edge count (see API and examples).
- **NAIVE**: Single linear equality; fewer variables, denser QUBO.


### Optional parameters

- **threshold**: For recursive networks (e.g. DIVIDE_AND_CONQUER), stop recursion when group size ≤ `threshold`.
- **var_prefix**: In the low-level API, optional prefix for auxiliary variables to avoid name collisions when merging QUBOs. See [Usage](usage.md).

## Low-level API

For direct access to QUBO or switching networks:

```python
from sparse_qubo.core.constraint import get_constraint_switches, ConstraintType
from sparse_qubo.core.network import NetworkType
from sparse_qubo.core.switch import switches_to_qubo

# Get Switches, then QUBO (variables, linear, quadratic, constant)
switches = get_constraint_switches(
    ["x0", "x1", "x2", "x3"],
    ConstraintType.ONE_HOT,
    NetworkType.DIVIDE_AND_CONQUER,
)
qubo = switches_to_qubo(switches)

print(qubo.variables, qubo.linear, qubo.quadratic, qubo.constant)
```

To work with Switch lists (e.g. for custom analysis or visualization):

```python
from sparse_qubo.core.constraint import ConstraintType, get_initial_nodes
from sparse_qubo.core.network import NetworkType
from sparse_qubo.core.switch import switches_to_qubo
from sparse_qubo.networks.divide_and_conquer_network import DivideAndConquerNetwork

variables = [f"x{i}" for i in range(4)]
left_nodes, right_nodes = get_initial_nodes(variables, ConstraintType.ONE_HOT)
switches = DivideAndConquerNetwork.generate_network(left_nodes, right_nodes)
# switches is list[Switch]; use switches_to_qubo(switches) to get QUBO
```

## Fixstars Amplify

For Amplify, use `sparse_qubo.create_constraint_amplify`. It accepts a list of `amplify.Variable` and returns an `amplify.Model`:

```python
import amplify
import sparse_qubo

variables = [amplify.Variable(f"x{i}") for i in range(4)]
model = sparse_qubo.create_constraint_amplify(
    variables,
    sparse_qubo.ConstraintType.ONE_HOT,
    sparse_qubo.NetworkType.DIVIDE_AND_CONQUER,
)
```

## Next steps

- [Examples](examples.md) — Inline examples and repository example overview
- [Usage](usage.md) — Constraint prefix counter and variable naming
- [API Reference](modules.md) — Full module and class reference
