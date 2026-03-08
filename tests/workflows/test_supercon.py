"""Tests for workflow helpers in ``aiida_epw.workflows.supercon``."""

from types import SimpleNamespace

import pytest

from aiida_epw.workflows.supercon import get_restart_parent_folder


def make_process_labelled_outputs(process_label, **outputs):
    """Create a lightweight object that mimics the outputs interface of a process node."""
    return SimpleNamespace(process_label=process_label, outputs=SimpleNamespace(**outputs))


def test_get_restart_parent_folder_prefers_stashed_outputs():
    """Prefer persistent restart outputs over transient remote folders."""
    process = make_process_labelled_outputs(
        "EpwPrepWorkChain",
        epw_folder="stash-folder",
        remote_stash="calc-stash",
        remote_folder="calc-remote",
    )

    assert get_restart_parent_folder(process) == "stash-folder"


def test_get_restart_parent_folder_falls_back_to_calc_outputs():
    """Fallback to the calcjob stash and then to its remote folder."""
    stashed = make_process_labelled_outputs("EpwBaseWorkChain", remote_stash="calc-stash")
    remote_only = make_process_labelled_outputs(
        "EpwBaseWorkChain", remote_folder="calc-remote"
    )

    assert get_restart_parent_folder(stashed) == "calc-stash"
    assert get_restart_parent_folder(remote_only) == "calc-remote"


def test_get_restart_parent_folder_requires_exposed_restart_data():
    """Raise a clear error if no restart-capable output is exposed."""
    process = make_process_labelled_outputs("EpwBaseWorkChain")

    with pytest.raises(ValueError, match="Could not determine a restart folder"):
        get_restart_parent_folder(process)
