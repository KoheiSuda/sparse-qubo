from collections import defaultdict
from math import ceil

from ..base_network import ISwitchingNetwork
from ..networks.bubble_sort_network import BubbleSortNetwork
from ..node import NodeAttribute, VariableNode
from ..permutation_channel import PermutationChannel


# Divide up and down, and apply BubbleSortNetwork when it becomes one-hot
class DivideAndConquerNetwork(ISwitchingNetwork):
    @classmethod
    def _generate_original_network(
        cls,
        left_nodes: list[VariableNode],
        right_nodes: list[VariableNode],
        threshold: int | None = None,
        reverse: bool = False,
    ) -> list[PermutationChannel]:
        if len(left_nodes) != len(right_nodes):
            raise ValueError("left_nodes and right_nodes must have the same length")

        num_variables: int = len(left_nodes)

        node_dict: defaultdict[NodeAttribute, list[VariableNode]] = defaultdict(list)
        for node in right_nodes:
            node_dict[node.attribute].append(node)

        # TODO: Currently only supports equal to
        if len(node_dict[NodeAttribute.ZERO_OR_ONE]) != 0:
            raise ValueError("ZERO_OR_ONE nodes are not supported")
        if len(node_dict[NodeAttribute.NOT_CARE]) != 0:
            raise ValueError("NOT_CARE nodes are not supported")
        if len(node_dict[NodeAttribute.ALWAYS_ZERO]) + len(node_dict[NodeAttribute.ALWAYS_ONE]) != num_variables:
            raise ValueError("right_nodes must consist only of ALWAYS_ZERO and ALWAYS_ONE nodes")
        if not all(node.attribute == NodeAttribute.ZERO_OR_ONE for node in left_nodes):
            raise ValueError("All left_nodes must have ZERO_OR_ONE attribute")

        perm_channels: list[PermutationChannel] = []

        if (
            len(node_dict[NodeAttribute.ALWAYS_ZERO]) == num_variables
            or len(node_dict[NodeAttribute.ALWAYS_ONE]) == num_variables
        ):
            perm_channels.extend([
                PermutationChannel(
                    left_nodes=frozenset([left_node.name]),
                    right_nodes=frozenset([right_node.name]),
                )
                for left_node, right_node in zip(left_nodes, right_nodes, strict=True)
            ])
            return perm_channels

        # Case of one-hot
        if len(node_dict[NodeAttribute.ALWAYS_ONE]) == 1:
            perm_channels.extend(
                BubbleSortNetwork._generate_original_network(
                    left_nodes,
                    node_dict[NodeAttribute.ALWAYS_ZERO] + node_dict[NodeAttribute.ALWAYS_ONE],
                )
            )
            return perm_channels
        elif len(node_dict[NodeAttribute.ALWAYS_ZERO]) == 1:
            perm_channels.extend(
                BubbleSortNetwork._generate_original_network(
                    left_nodes,
                    node_dict[NodeAttribute.ALWAYS_ONE] + node_dict[NodeAttribute.ALWAYS_ZERO],
                )
            )
            return perm_channels
        # Other cases
        else:
            if threshold is not None and num_variables <= threshold:
                perm_channels.append(
                    PermutationChannel(
                        left_nodes=frozenset([left_node.name for left_node in left_nodes]),
                        right_nodes=frozenset([right_node.name for right_node in right_nodes]),
                    )
                )
                return perm_channels

            aux_nodes: list[VariableNode] = [
                VariableNode(name=f"{left_node.name}_{idx}", attribute=NodeAttribute.ZERO_OR_ONE)
                for idx, left_node in enumerate(left_nodes)
            ]
            for i in range(num_variables // 2):
                perm_channels.append(
                    PermutationChannel(
                        left_nodes=frozenset([
                            left_nodes[i].name,
                            left_nodes[i + ceil(num_variables / 2)].name,
                        ]),
                        right_nodes=frozenset(
                            [
                                aux_nodes[i].name,
                                aux_nodes[i + ceil(num_variables / 2)].name,
                            ],
                        ),
                    )
                )
            # For nodes where no swap occurred, assign directly to aux_nodes
            if num_variables % 2 == 1:
                aux_nodes[num_variables // 2] = left_nodes[num_variables // 2]

            perm_channels.extend(
                cls._generate_original_network(
                    left_nodes=aux_nodes[: ceil(num_variables / 2)],
                    right_nodes=node_dict[NodeAttribute.ALWAYS_ONE][
                        : ceil(len(node_dict[NodeAttribute.ALWAYS_ONE]) / 2)
                    ]
                    + node_dict[NodeAttribute.ALWAYS_ZERO][
                        : ceil(num_variables / 2) - ceil(len(node_dict[NodeAttribute.ALWAYS_ONE]) / 2)
                    ],
                    threshold=threshold,
                )
            )
            perm_channels.extend(
                cls._generate_original_network(
                    left_nodes=aux_nodes[ceil(num_variables / 2) :],
                    right_nodes=node_dict[NodeAttribute.ALWAYS_ONE][
                        ceil(len(node_dict[NodeAttribute.ALWAYS_ONE]) / 2) :
                    ]
                    + node_dict[NodeAttribute.ALWAYS_ZERO][
                        ceil(num_variables / 2) - ceil(len(node_dict[NodeAttribute.ALWAYS_ONE]) / 2) :
                    ],
                    threshold=threshold,
                )
            )
            return perm_channels
