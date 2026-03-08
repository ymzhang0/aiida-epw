from types import SimpleNamespace

import pytest

from aiida_epw.tools.workchain import (
    find_related_calculation,
    get_parent_folder_calculation,
)


def test_get_parent_folder_calculation_returns_direct_creator():
    """Test that direct parent-folder creators are returned unchanged."""
    creator = SimpleNamespace(process_label="PhCalculation")
    parent_folder = SimpleNamespace(creator=creator)

    assert get_parent_folder_calculation(parent_folder) is creator


def test_get_parent_folder_calculation_unwraps_move_stash():
    """Test that stashed parent folders resolve to the original calculation."""
    creator = SimpleNamespace(process_label="PhCalculation")
    move_stash = SimpleNamespace(
        process_label="move_stash",
        inputs=SimpleNamespace(stash_data=SimpleNamespace(creator=creator)),
    )
    parent_folder = SimpleNamespace(creator=move_stash)

    assert get_parent_folder_calculation(parent_folder) is creator


def test_get_parent_folder_calculation_requires_creator():
    """Test that helper fails clearly for folders without provenance."""
    parent_folder = SimpleNamespace(creator=None)

    with pytest.raises(ValueError, match="does not have a creator"):
        get_parent_folder_calculation(parent_folder)


def test_find_related_calculation_rejects_non_epw_creator():
    """Test that EPW-specific helper still validates the resolved process label."""
    creator = SimpleNamespace(process_label="PhCalculation")
    parent_folder = SimpleNamespace(creator=creator)

    with pytest.raises(ValueError, match="not a valid epw calculation"):
        find_related_calculation(parent_folder)
