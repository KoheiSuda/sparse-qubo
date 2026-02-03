# API Reference

This page lists the main modules and classes of `sparse-qubo`.

**Public API (top-level package):** Use `sparse_qubo.create_constraint_dwave` for D-Wave/dimod BQMs and `sparse_qubo.create_constraint_amplify` for Fixstars Amplify models. Constraint and network types are `sparse_qubo.ConstraintType` and `sparse_qubo.NetworkType`. The sections below document the implementation modules.

## Core modules

::: sparse_qubo.core.constraint
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.core.network
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.core.node
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.core.switch
    options:
      show_root_heading: true
      show_source: true

## Network implementations

::: sparse_qubo.networks.benes_network
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.bitonic_sort_network
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.bubble_sort_network
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.clique_network
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.clos_network_base
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.clos_network_max_degree
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.clos_network_minimum_edge
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.divide_and_conquer_network
    options:
      show_root_heading: true
      show_source: true

::: sparse_qubo.networks.oddeven_merge_sort_network
    options:
      show_root_heading: true
      show_source: true

## D-Wave integration

Public entry point: `sparse_qubo.create_constraint_dwave`. Implementation module:

::: sparse_qubo.dwave.constraint
    options:
      show_root_heading: true
      show_source: true

## Fixstars Amplify integration

Public entry point: `sparse_qubo.create_constraint_amplify`. Implementation module:

::: sparse_qubo.fixstars_amplify.constraint
    options:
      show_root_heading: true
      show_source: true
