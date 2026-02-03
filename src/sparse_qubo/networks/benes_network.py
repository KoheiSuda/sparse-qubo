"""Implementation of Benes network"""

from sparse_qubo.core.switch import Switch
from sparse_qubo.networks.clos_network_base import ClosNetworkBase


class BenesNetwork(ClosNetworkBase):
    """Benes network implementation; requires power-of-2 variable count."""

    @classmethod
    def _implement_if_small(cls, left_nodes: list[str], right_nodes: list[str]) -> list[Switch] | None:
        N = max(len(left_nodes), len(right_nodes))
        if N <= 2:
            return [
                Switch(
                    left_nodes=frozenset(left_nodes),
                    right_nodes=frozenset(right_nodes),
                )
            ]
        return None

    @classmethod
    def _determine_switch_sizes(cls, N_left: int, N_right: int) -> tuple[int, int]:
        """Return (n, r) for Benes network with n*r >= N."""
        N = max(N_left, N_right)
        n, r = 2, 1
        while n * r < N:
            r *= 2
        return n, r
