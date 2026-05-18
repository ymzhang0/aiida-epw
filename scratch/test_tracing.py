import sys
from unittest.mock import MagicMock

# Mock AiiDA if needed, but here we can just mock the attributes
class MockNode:
    def __init__(self, creator=None):
        self.creator = creator

class MockProcess:
    def __init__(self, label, inputs=None):
        self.process_label = label
        self.inputs = MagicMock()
        if inputs:
            for k, v in inputs.items():
                setattr(self.inputs, k, v)

def get_parent_folder_calculation(parent_folder):
    """Return the calculation that produced a remote or stashed parent folder."""
    current_node = parent_folder

    while True:
        creator = current_node.creator
        if creator is None:
            raise ValueError(f"The provided node {current_node} does not have a creator.")

        if creator.process_label == "move_stash":
            current_node = creator.inputs.stash_data
        else:
            return creator

def test_tracing():
    # Setup the chain:
    # PhCalculation -> node1 -> move_stash (node2) -> move_stash (node3)
    
    ph_calc = MockProcess("PhCalculation")
    node1 = MockNode(creator=ph_calc)
    
    move1_inputs = {'stash_data': node1}
    move1 = MockProcess("move_stash", inputs=move1_inputs)
    node2 = MockNode(creator=move1)
    
    move2_inputs = {'stash_data': node2}
    move2 = MockProcess("move_stash", inputs=move2_inputs)
    node3 = MockNode(creator=move2)
    
    print("Testing 2-level move...")
    result = get_parent_folder_calculation(node3)
    assert result == ph_calc
    print("Success: Found PhCalculation")

    print("Testing 0-level move...")
    result = get_parent_folder_calculation(node1)
    assert result == ph_calc
    print("Success: Found PhCalculation directly")

if __name__ == "__main__":
    try:
        test_tracing()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
