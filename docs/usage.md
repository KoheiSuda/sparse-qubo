# Usage

## Constraint prefix counter

When you build constraint QUBOs with `get_constraint_qubo` or high-level APIs like `dwave.constraint()` and `fixstars_amplify.constraint()`, the library adds **auxiliary variables** (e.g. `L0`, `R0`, and variables created inside the switching network). To avoid name collisions when merging multiple constraint QUBOs into one BQM, each constraint’s auxiliary variables are given a unique **prefix** (e.g. `C0_L0`, `C1_L0`).

### How the counter behaves

- **When it increments**: Each time you create a constraint (with the default `var_prefix=None`), the internal counter is used as the prefix (`C0`, `C1`, `C2`, …) and then incremented.
- **When it resets**: The counter is **not** reset automatically. It is reset only in these cases:
  1. **Process start**: Restarting Python (e.g. re-running a script or restarting a Jupyter kernel) reloads the module, so the counter starts at 0 again.
  2. **Explicit reset**: You call `reset_constraint_prefix_counter()` from `sparse_qubo.core.constraint`.

### Practical impact

- **Same process, multiple constraints**: If you create several constraints and add their QUBOs/BQMs together, each gets a different prefix (`C0`, `C1`, …), so variable names do not collide.
- **Re-running a script or notebook (same kernel)**: The counter keeps increasing. For example, the second run might use `C3`, `C4`, `C5` instead of `C0`, `C1`, `C2`. Names stay unique; only the numeric part changes.
- **Starting a new model in the same process**: If you want the “first” constraint of a new model to use `C0` again, call `reset_constraint_prefix_counter()` before building the new model. This is optional and mainly useful for reproducible variable names or tests.
