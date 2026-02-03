"""Switch elements and QUBO conversion for switching networks.

A Switch encodes a constraint that the sum of left variables plus left_constant
equals the sum of right variables plus right_constant. Switch.to_qubo converts
a list of Switch elements into a single QUBO (variables, linear, quadratic, constant).
"""

import os
import re
from collections import defaultdict, deque
from itertools import combinations, product
from typing import Self

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, Field


class QUBO(BaseModel):
    """QUBO representation: variables, linear and quadratic coefficients, and constant."""

    variables: frozenset[str] = Field(description="Set of binary variables")
    quadratic: dict[frozenset[str], float] = Field(description="Coefficients between binary variables")
    linear: dict[str, float] = Field(description="Coefficients of binary variables")
    constant: float = Field(default=0, description="Constant term")


class Switch(BaseModel):
    """Single switch: left and right variable sets and optional integer constants.

    The constraint is: sum(left_nodes) + left_constant == sum(right_nodes) + right_constant.
    """

    left_nodes: frozenset[str] = Field(default_factory=frozenset, description="Binary variables (left side)")
    right_nodes: frozenset[str] = Field(default_factory=frozenset, description="Binary variables (right side)")
    left_constant: int = Field(default=0, description="Constant term (left side)")
    right_constant: int = Field(default=0, description="Constant term (right side)")

    def __post_init__(self) -> None:
        # Check for duplicate variables
        if len(self.left_nodes | self.right_nodes) != len(self.left_nodes) + len(self.right_nodes):
            raise ValueError("Duplicate variables found between left_nodes and right_nodes")

    def __repr__(self) -> str:
        return f"Switch(left={sorted(self.left_nodes)} + {self.left_constant}, right={sorted(self.right_nodes)} + {self.right_constant})"

    @property
    def num_variables(self) -> int:
        """Number of variables in this switch (left + right)."""
        return len(self.left_nodes) + len(self.right_nodes)

    @property
    def num_edges(self) -> int:
        """Number of quadratic terms if this switch were converted to QUBO alone."""
        return self.num_variables * (self.num_variables - 1) // 2

    @classmethod
    def to_qubo(cls, switches: list[Self]) -> QUBO:
        """Convert a list of Switch elements into a single QUBO (sum of (L + c_L - R - c_R)^2 terms)."""
        variables: set[str] = set()
        quadratic: dict[frozenset[str], float] = defaultdict(float)
        linear: dict[str, float] = defaultdict(float)
        constant: float = 0

        # Constant terms are not discarded
        # At the call stage, each node in Switch is either NOT_CARE or ZERO_OR_ONE
        # ALWAYS_ZERO and ALWAYS_ONE are processed as constant terms
        # NOT_CARE is treated the same as ZERO_OR_ONE
        for switch in switches:
            # (L1 + L2 - R1 - R2 + C)^2
            # = 2L1L2 + 2R1R2 - 2(L1R1 + ...)
            # + L1 + L2 + R1 + R2 + 2C(L1 + L2 - R1 - R2)
            # + C^2

            switch_constant = switch.left_constant - switch.right_constant
            variables.update(switch.left_nodes)
            variables.update(switch.right_nodes)
            # quadratic
            for node1, node2 in combinations(switch.left_nodes, 2):
                quadratic[frozenset((node1, node2))] += 2
            for node1, node2 in combinations(switch.right_nodes, 2):
                quadratic[frozenset((node1, node2))] += 2
            for node1, node2 in product(switch.left_nodes, switch.right_nodes):
                quadratic[frozenset((node1, node2))] -= 2
            # linear
            for node in switch.left_nodes:
                linear[node] += 2 * switch_constant
                linear[node] += 1  # Because x*x = x
            for node in switch.right_nodes:
                linear[node] -= 2 * switch_constant
                linear[node] += 1
            # constant
            constant += switch_constant**2
        qubo = QUBO(
            variables=frozenset(variables),
            quadratic=quadratic,
            linear=linear,
            constant=constant,
        )

        return qubo

    @classmethod
    def left_node_to_switch(cls, switches: list[Self]) -> dict[str, int]:
        """Map each left-side variable name to the index of the switch that contains it."""
        left_node_to_switch: dict[str, int] = {}
        for idx, switch in enumerate(switches):
            for node in switch.left_nodes:
                left_node_to_switch[node] = idx
        return left_node_to_switch

    @classmethod
    def right_node_to_switch(cls, switches: list[Self]) -> dict[str, int]:
        """Map each right-side variable name to the index of the switch that contains it."""
        right_node_to_switch: dict[str, int] = {}
        for idx, switch in enumerate(switches):
            for node in switch.right_nodes:
                right_node_to_switch[node] = idx
        return right_node_to_switch

    @classmethod
    def determine_layer_structure(
        cls,
        switches: list[Self],
    ) -> dict[int, list[int]]:
        """Determine which layer each switch belongs to. Returns {layer_number: [switch_indices]} (0-indexed)."""
        left_node_to_switch = cls.left_node_to_switch(switches)
        layer_structure: dict[int, list[int]] = {}
        switch_idx_to_layer_number: dict[int, int] = {}

        # Identify original variables (L0, L1, ...)
        waiting_left_nodes: deque[tuple[str, int]] = deque()  # (node, layer_number)
        for switch in switches:
            for node in switch.left_nodes:
                if re.match(r"^L\d+$", node):
                    waiting_left_nodes.append((node, 0))

        # Determine the layer of each switch
        while waiting_left_nodes:
            node, layer_number = waiting_left_nodes.pop()
            if node not in left_node_to_switch:
                continue
            if left_node_to_switch[node] in switch_idx_to_layer_number:
                continue
            switch_idx_to_layer_number[left_node_to_switch[node]] = layer_number
            waiting_left_nodes.extend(
                (node, layer_number + 1) for node in switches[left_node_to_switch[node]].right_nodes
            )

        for switch_idx, layer_number in switch_idx_to_layer_number.items():
            if layer_number not in layer_structure:
                layer_structure[layer_number] = []
            layer_structure[layer_number].append(switch_idx)

        return layer_structure

    @classmethod
    def visualize_switches(
        cls,
        switches: list["Switch"],
        output_path: str,
        layout_type: str = "network",  # "network" or "spring"
        layer_spacing: float = 2.0,
        node_spacing: float = 1.0,
    ) -> None:
        """Draw the relationship between variables and switches as a graph and save to output_path."""
        all_nodes: set[str] = set()
        for switch in switches:
            all_nodes.update(switch.left_nodes)
            all_nodes.update(switch.right_nodes)

        # Generate colors for each switch (same as generate_and_draw_qubo_network)
        cmap = plt.get_cmap("tab10")
        switch_colors = [cmap(i % 10) for i in range(len(switches))]

        graph: nx.Graph[str] = nx.Graph()
        graph.add_nodes_from(all_nodes, node_color="blue")

        # Set colors for switch nodes
        for switch_idx in range(len(switches)):
            graph.add_node("C" + str(switch_idx), node_color=switch_colors[switch_idx])

        # Add constant nodes and set colors for edges
        for switch_idx, switch in enumerate(switches):
            switch_color = switch_colors[switch_idx]
            for node in switch.left_nodes:
                graph.add_edge(node, "C" + str(switch_idx), edge_color=switch_color)
            for node in switch.right_nodes:
                graph.add_edge(node, "C" + str(switch_idx), edge_color=switch_color)
            if switch.left_constant != 0:
                const_node = f"LC_{switch_idx}_{switch.left_constant}"
                graph.add_node(const_node, node_color="g")
                graph.add_edge(
                    const_node,
                    "C" + str(switch_idx),
                    edge_color=switch_color,
                )
            if switch.right_constant != 0:
                const_node = f"RC_{switch_idx}_{switch.right_constant}"
                graph.add_node(const_node, node_color="green")
                graph.add_edge(
                    "C" + str(switch_idx),
                    const_node,
                    edge_color=switch_color,
                )

        # Determine layout
        pos: dict[str, tuple[float, float]]
        if layout_type == "network":
            pos = cls._create_network_layout_with_layers(
                switches,
                all_nodes,
                layer_spacing,
                node_spacing,
            )
        else:
            raw_pos = nx.spring_layout(graph)
            pos = {node: (float(p[0]), float(p[1])) for node, p in raw_pos.items()}

        edge_colors: list[str] = [graph[u][v]["edge_color"] for u, v in graph.edges()]

        # Draw the graph
        plt.figure(figsize=(max(12, len(switches) * 1.5), 10))

        # Draw nodes by type
        # Blue nodes (circle)
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=list(all_nodes),
            node_color="blue",
            node_size=1500,
            node_shape="o",
            alpha=0.5,
        )
        # switch nodes (square, different color for each switch)
        for switch_idx in range(len(switches)):
            switch_node = "C" + str(switch_idx)
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[switch_node],
                node_color=[switch_colors[switch_idx]],
                node_size=5000,
                node_shape="s",
                alpha=0.5,
            )
        # Green constant nodes (circle)
        const_nodes = [node for node in graph.nodes() if node.startswith("LC_") or node.startswith("RC_")]
        if const_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=const_nodes,
                node_color="green",
                node_size=1500,
                node_shape="o",
                alpha=0.5,
            )

        # Draw edges and labels
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=edge_colors,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=12,
            font_weight="bold",
        )

        # Display layer information
        if layout_type == "network":
            layer_structure = cls.determine_layer_structure(switches)
            cls._add_layer_labels(layer_structure, layer_spacing)
            plt.title(
                f"Switches Network ({len(layer_structure)} layers)",
                fontsize=16,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "switches.png"), dpi=300, bbox_inches="tight")
        plt.close()

    @classmethod
    def _create_network_layout_with_layers(
        cls,
        switches: list["Switch"],
        all_nodes: set[str],
        layer_spacing: float,
        node_spacing: float,
    ) -> dict[str, tuple[float, float]]:
        """
        Create a manual layout based on layer structure
        Layers are arranged to the right, and switches within each layer are arranged vertically
        """
        pos = {}

        # Determine layer structure
        layer_structure = cls.determine_layer_structure(switches)

        # Calculate position of each layer
        for layer_num, switch_indices in sorted(layer_structure.items()):
            x = (layer_num - 1) * layer_spacing

            # Arrange switches vertically within each layer
            for i, switch_idx in enumerate(switch_indices):
                switch = switches[switch_idx]
                switch_y = -(i - len(switch_indices) / 2) * node_spacing

                # Place switch node
                pos[f"C{switch_idx}"] = (x, switch_y)

                # Place left nodes on the left (smaller index is higher, center aligns with switch node)
                left_x_start = x - 0.8
                sorted_left_nodes = sorted(switch.left_nodes)
                for j, node in enumerate(sorted_left_nodes):
                    y = switch_y - (j - (len(sorted_left_nodes) - 1) / 2) * 0.3
                    pos[node] = (left_x_start, y)

                # Place right nodes on the right (smaller index is higher, center aligns with switch node)
                right_x_start = x + 0.8
                sorted_right_nodes = sorted(switch.right_nodes)
                for j, node in enumerate(sorted_right_nodes):
                    y = switch_y - (j - (len(sorted_right_nodes) - 1) / 2) * 0.3
                    pos[node] = (right_x_start, y)

                # Place constant nodes (same height as switch node)
                if switch.left_constant != 0:
                    const_node = f"LC_{switch_idx}_{switch.left_constant}"
                    pos[const_node] = (left_x_start, switch_y)
                if switch.right_constant != 0:
                    const_node = f"RC_{switch_idx}_{switch.right_constant}"
                    pos[const_node] = (right_x_start, switch_y)

        # Place other nodes (unconnected nodes) in appropriate positions
        used_nodes = set(pos.keys())
        unused_nodes = all_nodes - used_nodes

        if unused_nodes:
            # Place unused nodes to the right of the last layer
            max_layer = max(layer_structure.keys()) if layer_structure else 1
            last_x = (max_layer - 1) * layer_spacing + 1.5
            for i, node in enumerate(sorted(unused_nodes)):
                pos[node] = (last_x, -2.0 - i * node_spacing)

        return pos

    @classmethod
    def _add_layer_labels(
        cls,
        layer_structure: dict[int, list[int]],
        layer_spacing: float,
    ) -> None:
        """
        Add layer labels
        """
        for layer_num, _ in layer_structure.items():
            x = (layer_num - 1) * layer_spacing
            # Place layer labels at the top of each layer
            plt.text(
                x,
                3.0,
                f"Layer {layer_num}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
            )
