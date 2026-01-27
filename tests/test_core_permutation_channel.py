from sparse_qubo.core.permutation_channel import QUBO, PermutationChannel


class TestPermutationChannel:
    """Tests for PermutationChannel model."""

    def test_permutation_channel_creation(self) -> None:
        """Test creating a basic PermutationChannel."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        assert channel.left_nodes == frozenset(["L0", "L1"])
        assert channel.right_nodes == frozenset(["R0", "R1"])
        assert channel.left_constant == 0
        assert channel.right_constant == 0

    def test_permutation_channel_with_constants(self) -> None:
        """Test creating PermutationChannel with constants."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0"]),
            right_nodes=frozenset(["R0"]),
            left_constant=1,
            right_constant=2,
        )
        assert channel.left_constant == 1
        assert channel.right_constant == 2

    def test_permutation_channel_num_variables(self) -> None:
        """Test num_variables property."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0", "L1", "L2"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        assert channel.num_variables == 5

    def test_permutation_channel_num_edges(self) -> None:
        """Test num_edges property."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        # 4 variables -> 4 * 3 / 2 = 6 edges
        assert channel.num_edges == 6

    def test_permutation_channel_repr(self) -> None:
        """Test PermutationChannel string representation."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0"]),
            left_constant=1,
            right_constant=2,
        )
        repr_str = repr(channel)
        assert "L0" in repr_str or "L1" in repr_str
        assert "R0" in repr_str
        assert "1" in repr_str  # left_constant
        assert "2" in repr_str  # right_constant


class TestPermutationChannelToQUBO:
    """Tests for PermutationChannel.to_qubo method."""

    def test_single_channel_to_qubo(self) -> None:
        """Test converting a single PermutationChannel to QUBO."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        qubo = PermutationChannel.to_qubo([channel])

        # Check variables
        assert qubo.variables == frozenset(["L0", "L1", "R0", "R1"])

        # Check quadratic terms: 2L0L1 + 2R0R1 - 2(L0R0 + L0R1 + L1R0 + L1R1)
        assert qubo.quadratic[frozenset(["L0", "L1"])] == 2
        assert qubo.quadratic[frozenset(["R0", "R1"])] == 2
        assert qubo.quadratic[frozenset(["L0", "R0"])] == -2
        assert qubo.quadratic[frozenset(["L0", "R1"])] == -2
        assert qubo.quadratic[frozenset(["L1", "R0"])] == -2
        assert qubo.quadratic[frozenset(["L1", "R1"])] == -2

        # Check linear terms: each variable has coefficient 1 (from x*x = x)
        assert qubo.linear["L0"] == 1
        assert qubo.linear["L1"] == 1
        assert qubo.linear["R0"] == 1
        assert qubo.linear["R1"] == 1

        # Check constant: (0 - 0)^2 = 0
        assert qubo.constant == 0

    def test_channel_with_constants_to_qubo(self) -> None:
        """Test converting PermutationChannel with constants to QUBO."""
        channel = PermutationChannel(
            left_nodes=frozenset(["L0"]),
            right_nodes=frozenset(["R0"]),
            left_constant=1,
            right_constant=2,
        )
        qubo = PermutationChannel.to_qubo([channel])

        # channel_constant = 1 - 2 = -1
        # Linear terms: L0 has 2*(-1) + 1 = -1, R0 has -2*(-1) + 1 = 3
        assert qubo.linear["L0"] == -1
        assert qubo.linear["R0"] == 3

        # Constant: (-1)^2 = 1
        assert qubo.constant == 1

    def test_multiple_channels_to_qubo(self) -> None:
        """Test converting multiple PermutationChannels to QUBO."""
        channel1 = PermutationChannel(
            left_nodes=frozenset(["L0"]),
            right_nodes=frozenset(["R0"]),
        )
        channel2 = PermutationChannel(
            left_nodes=frozenset(["L1"]),
            right_nodes=frozenset(["R1"]),
        )
        qubo = PermutationChannel.to_qubo([channel1, channel2])

        assert qubo.variables == frozenset(["L0", "L1", "R0", "R1"])
        # Each channel contributes independently
        assert qubo.linear["L0"] == 1
        assert qubo.linear["L1"] == 1
        assert qubo.linear["R0"] == 1
        assert qubo.linear["R1"] == 1


class TestPermutationChannelHelpers:
    """Tests for PermutationChannel helper methods."""

    def test_left_node_to_channel(self) -> None:
        """Test left_node_to_channel mapping."""
        channels = [
            PermutationChannel(left_nodes=frozenset(["L0", "L1"]), right_nodes=frozenset(["R0"])),
            PermutationChannel(left_nodes=frozenset(["L2"]), right_nodes=frozenset(["R1"])),
        ]
        mapping = PermutationChannel.left_node_to_channel(channels)
        assert mapping["L0"] == 0
        assert mapping["L1"] == 0
        assert mapping["L2"] == 1

    def test_right_node_to_channel(self) -> None:
        """Test right_node_to_channel mapping."""
        channels = [
            PermutationChannel(left_nodes=frozenset(["L0"]), right_nodes=frozenset(["R0", "R1"])),
            PermutationChannel(left_nodes=frozenset(["L1"]), right_nodes=frozenset(["R2"])),
        ]
        mapping = PermutationChannel.right_node_to_channel(channels)
        assert mapping["R0"] == 0
        assert mapping["R1"] == 0
        assert mapping["R2"] == 1

    def test_determine_layer_structure(self) -> None:
        """Test determine_layer_structure method."""
        # Create a simple two-layer network
        channels = [
            PermutationChannel(left_nodes=frozenset(["L0", "L1"]), right_nodes=frozenset(["M0", "M1"])),
            PermutationChannel(left_nodes=frozenset(["M0", "M1"]), right_nodes=frozenset(["R0", "R1"])),
        ]
        layer_structure = PermutationChannel.determine_layer_structure(channels)
        assert 0 in layer_structure
        assert 1 in layer_structure
        assert len(layer_structure[0]) == 1
        assert len(layer_structure[1]) == 1


class TestQUBO:
    """Tests for QUBO model."""

    def test_qubo_creation(self) -> None:
        """Test creating a QUBO."""
        qubo = QUBO(
            variables=frozenset(["x0", "x1"]),
            quadratic={frozenset(["x0", "x1"]): 2.0},
            linear={"x0": 1.0, "x1": -1.0},
            constant=0.5,
        )
        assert qubo.variables == frozenset(["x0", "x1"])
        assert qubo.quadratic[frozenset(["x0", "x1"])] == 2.0
        assert qubo.linear["x0"] == 1.0
        assert qubo.linear["x1"] == -1.0
        assert qubo.constant == 0.5

    def test_qubo_default_constant(self) -> None:
        """Test QUBO with default constant value."""
        qubo = QUBO(
            variables=frozenset(["x0"]),
            quadratic={},
            linear={"x0": 1.0},
        )
        assert qubo.constant == 0
