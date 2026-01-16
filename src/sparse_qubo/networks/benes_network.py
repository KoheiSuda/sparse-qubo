from ..permutation_channel import PermutationChannel
from .clos_network_base import ClosNetworkBase


class BenesNetwork(ClosNetworkBase):
    # Return implementation for small cases
    @classmethod
    def _implement_if_small(cls, left_nodes: list[str], right_nodes: list[str]) -> list[PermutationChannel] | None:
        N = max(len(left_nodes), len(right_nodes))
        if N <= 2:
            return [
                PermutationChannel(
                    left_nodes=frozenset(left_nodes),
                    right_nodes=frozenset(right_nodes),
                )
            ]
        return None

    # Determine the values of n and r
    @classmethod
    def _determine_channel_sizes(cls, N_left: int, N_right: int) -> tuple[int, int]:
        N = max(N_left, N_right)
        n, r = 2, 1
        while n * r < N:
            r *= 2
        return n, r
