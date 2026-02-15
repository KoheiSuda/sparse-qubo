from sparse_qubo.core.switch import QUBO, Switch, switches_to_qubo


class TestSwitch:
    """Tests for Switch model."""

    def test_switch_creation(self) -> None:
        """Test creating a basic Switch."""
        switch = Switch(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        assert switch.left_nodes == frozenset(["L0", "L1"])
        assert switch.right_nodes == frozenset(["R0", "R1"])
        assert switch.left_constant == 0
        assert switch.right_constant == 0

    def test_switch_with_constants(self) -> None:
        """Test creating Switch with constants."""
        switch = Switch(
            left_nodes=frozenset(["L0"]),
            right_nodes=frozenset(["R0"]),
            left_constant=1,
            right_constant=2,
        )
        assert switch.left_constant == 1
        assert switch.right_constant == 2

    def test_switch_num_variables(self) -> None:
        """Test num_variables property."""
        switch = Switch(
            left_nodes=frozenset(["L0", "L1", "L2"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        assert switch.num_variables == 5

    def test_switch_num_edges(self) -> None:
        """Test num_edges property."""
        switch = Switch(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        # 4 variables -> 4 * 3 / 2 = 6 edges
        assert switch.num_edges == 6

    def test_switch_repr(self) -> None:
        """Test Switch string representation."""
        switch = Switch(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0"]),
            left_constant=1,
            right_constant=2,
        )
        repr_str = repr(switch)
        assert "L0" in repr_str or "L1" in repr_str
        assert "R0" in repr_str
        assert "1" in repr_str  # left_constant
        assert "2" in repr_str  # right_constant


class TestSwitchToQUBO:
    """Tests for switches_to_qubo function."""

    def test_single_switch_to_qubo(self) -> None:
        """Test converting a single Switch to QUBO."""
        switch = Switch(
            left_nodes=frozenset(["L0", "L1"]),
            right_nodes=frozenset(["R0", "R1"]),
        )
        qubo = switches_to_qubo([switch])

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

    def test_switch_with_constants_to_qubo(self) -> None:
        """Test converting Switch with constants to QUBO."""
        switch = Switch(
            left_nodes=frozenset(["L0"]),
            right_nodes=frozenset(["R0"]),
            left_constant=1,
            right_constant=2,
        )
        qubo = switches_to_qubo([switch])

        # switch_constant = 1 - 2 = -1
        # Linear terms: L0 has 2*(-1) + 1 = -1, R0 has -2*(-1) + 1 = 3
        assert qubo.linear["L0"] == -1
        assert qubo.linear["R0"] == 3

        # Constant: (-1)^2 = 1
        assert qubo.constant == 1

    def test_multiple_switches_to_qubo(self) -> None:
        """Test converting multiple Switches to QUBO."""
        switch1 = Switch(
            left_nodes=frozenset(["L0"]),
            right_nodes=frozenset(["R0"]),
        )
        switch2 = Switch(
            left_nodes=frozenset(["L1"]),
            right_nodes=frozenset(["R1"]),
        )
        qubo = switches_to_qubo([switch1, switch2])

        assert qubo.variables == frozenset(["L0", "L1", "R0", "R1"])
        # Each switch contributes independently
        assert qubo.linear["L0"] == 1
        assert qubo.linear["L1"] == 1
        assert qubo.linear["R0"] == 1
        assert qubo.linear["R1"] == 1


class TestSwitchHelpers:
    """Tests for Switch helper methods."""

    def test_left_node_to_switch(self) -> None:
        """Test left_node_to_switch mapping."""
        switches = [
            Switch(left_nodes=frozenset(["L0", "L1"]), right_nodes=frozenset(["R0"])),
            Switch(left_nodes=frozenset(["L2"]), right_nodes=frozenset(["R1"])),
        ]
        mapping = Switch.left_node_to_switch(switches)
        assert mapping["L0"] == 0
        assert mapping["L1"] == 0
        assert mapping["L2"] == 1

    def test_right_node_to_switch(self) -> None:
        """Test right_node_to_switch mapping."""
        switches = [
            Switch(left_nodes=frozenset(["L0"]), right_nodes=frozenset(["R0", "R1"])),
            Switch(left_nodes=frozenset(["L1"]), right_nodes=frozenset(["R2"])),
        ]
        mapping = Switch.right_node_to_switch(switches)
        assert mapping["R0"] == 0
        assert mapping["R1"] == 0
        assert mapping["R2"] == 1

    def test_determine_layer_structure(self) -> None:
        """Test determine_layer_structure method."""
        # Create a simple two-layer network
        switches = [
            Switch(left_nodes=frozenset(["L0", "L1"]), right_nodes=frozenset(["M0", "M1"])),
            Switch(left_nodes=frozenset(["M0", "M1"]), right_nodes=frozenset(["R0", "R1"])),
        ]
        layer_structure = Switch.determine_layer_structure(switches)
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
