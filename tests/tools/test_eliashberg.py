"""Tests for adaptive Eliashberg sampling helpers."""

from aiida_epw.tools.eliashberg import (
    get_temperature_list,
    set_temperature_list,
    suggest_temperatures,
)


def test_temperature_list_normalizes_epw_range():
    """Test EPW range-style temperatures are converted to explicit lists."""
    parameters = {"INPUTEPW": {"temps": "1.0 5.0", "nstemp": 3}}

    assert get_temperature_list(parameters) == [1.0, 3.0, 5.0]

    updated = set_temperature_list(parameters, [1.0, 2.0, 3.0])

    assert updated["INPUTEPW"]["temps"] == [1.0, 2.0, 3.0]
    assert "nstemp" not in updated["INPUTEPW"]


def test_suggest_temperatures_expands_high_when_gap_remains_open():
    """Test high-temperature expansion when the gap has not closed."""
    report = suggest_temperatures(
        [1.0, 2.0, 3.0],
        {"T": [1.0, 2.0, 3.0], "gap": [1.0, 0.8, 0.6]},
    )

    assert report["status"] == "RANGE_TOO_LOW"
    assert max(report["temperatures"]) > 3.0
    updated = set_temperature_list({"INPUTEPW": {}}, report["temperatures"])
    assert "nstemp" not in updated["INPUTEPW"]


def test_suggest_temperatures_expands_low_when_gap_is_closed():
    """Test low-temperature expansion when all gaps are effectively closed."""
    report = suggest_temperatures(
        [4.0, 5.0, 6.0],
        {"T": [4.0, 5.0, 6.0], "gap": [0.0, 0.0, 0.0]},
    )

    assert report["status"] == "RANGE_TOO_HIGH"
    assert min(report["temperatures"]) < 4.0


def test_suggest_temperatures_refines_bracketed_gap_closing():
    """Test adding temperatures in a bracketed gap-closing region."""
    report = suggest_temperatures(
        [1.0, 2.0, 3.0],
        {"T": [1.0, 2.0, 3.0], "gap": [1.0, 0.3, 0.01]},
        refine_points=5,
    )

    assert report["status"] == "NEED_MORE_POINTS"
    assert len(report["temperatures"]) > 3
    assert report["temperatures"][0] == 1.0
    assert report["temperatures"][-1] == 3.0


def test_suggest_temperatures_accepts_multigap_series():
    """Test representative gaps are extracted from anisotropic multi-gap values."""
    report = suggest_temperatures(
        [1.0, 2.0, 3.0],
        {"T": [1.0, 2.0, 3.0], "gap": [[1.0, 1.2], [0.4, 0.5], [0.01]]},
    )

    assert report["status"] in {"GOOD", "NEED_MORE_POINTS"}
