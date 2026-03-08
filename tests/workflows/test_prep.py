"""Tests for workflow helpers in ``aiida_epw.workflows.prep``."""

from aiida_epw.workflows.prep import (
    should_run_bands_interpolation,
    validate_inputs,
)


def test_validate_inputs_requires_w90_bands():
    """The preparation workflow currently requires the Wannier90 step."""
    message = validate_inputs({"ph_base": {}, "epw_base": {}, "epw_bands": {}})

    assert message == (
        "`w90_bands` inputs are required because this work chain needs the "
        "NSCF and Wannier checkpoint folders produced by the Wannier90 step."
    )


def test_validate_inputs_allows_skipping_band_interpolation_namespace():
    """The bands namespace is optional when the interpolation step is disabled."""
    message = validate_inputs(
        {
            "w90_bands": {},
            "ph_base": {},
            "epw_base": {},
            "do_bands_interpolation": False,
        }
    )

    assert message is None


def test_should_run_bands_interpolation_respects_flag():
    """The interpolation step should only run when explicitly enabled."""
    assert should_run_bands_interpolation({"do_bands_interpolation": True, "epw_bands": {}})
    assert not should_run_bands_interpolation(
        {"do_bands_interpolation": False, "epw_bands": {}}
    )
    assert not should_run_bands_interpolation({"do_bands_interpolation": True})
