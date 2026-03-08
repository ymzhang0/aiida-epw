"""Tests for workflow helpers in ``aiida_epw.workflows.prep``."""

from pathlib import Path
from types import SimpleNamespace

from aiida import orm

from aiida_epw.workflows.prep import (
    EpwPrepWorkChain,
    get_target_basepath,
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


def test_get_target_basepath_uses_local_workdir(fixture_localhost):
    """The stash basepath should be derived from the computer work directory."""
    assert get_target_basepath(fixture_localhost) == Path(
        fixture_localhost.get_workdir(), "stash"
    ).as_posix()


def test_should_run_wannier90_depends_on_namespace():
    """The Wannier90 step runs only when its namespace is present."""
    assert EpwPrepWorkChain.should_run_wannier90(
        SimpleNamespace(inputs={"w90_bands": {}})
    )
    assert not EpwPrepWorkChain.should_run_wannier90(SimpleNamespace(inputs={}))


def test_should_run_epw_bands_delegates_to_helper():
    """The workchain method should follow the same interpolation gating helper."""
    process = SimpleNamespace(
        inputs={
            "do_bands_interpolation": orm.Bool(True),
            "epw_bands": {},
        }
    )

    assert EpwPrepWorkChain.should_run_epw_bands(process)


def test_get_builder_from_protocol_skips_epw_bands_when_disabled(
    fixture_code,
    fixture_localhost,
    generate_remote_data,
    generate_structure,
    monkeypatch,
):
    """The prep protocol builder should omit the optional bands namespace when disabled."""
    from aiida.common import AttributeDict

    epw_code = fixture_code("epw.epw")

    def fake_w90_builder(*args, **kwargs):
        builder = AttributeDict()
        builder.structure = generate_structure()
        builder.open_grid = {}
        builder.projwfc = {}
        return builder

    def fake_ph_builder(*args, **kwargs):
        builder = AttributeDict()
        builder.clean_workdir = orm.Bool(False)
        builder.qpoints_distance = orm.Float(0.4)
        return builder

    def fake_epw_builder(*args, **kwargs):
        builder = AttributeDict()
        builder.code = epw_code
        builder.parameters = orm.Dict({"INPUTEPW": {}})
        builder.options = orm.Dict(
            {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                },
                "max_wallclock_seconds": 1800,
                "withmpi": True,
            }
        )
        builder.qfpoints_distance = orm.Float(0.1)
        builder.kfpoints_factor = orm.Int(2)
        builder.max_iterations = orm.Int(2)
        return builder

    monkeypatch.setattr(
        EpwPrepWorkChain,
        "get_builder",
        classmethod(lambda cls: AttributeDict()),
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.Wannier90BandsWorkChain.get_builder_from_protocol",
        fake_w90_builder,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.Wannier90OptimizeWorkChain.get_builder_from_protocol",
        fake_w90_builder,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.PhBaseWorkChain.get_builder_from_protocol",
        fake_ph_builder,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.EpwBaseWorkChain.get_builder_from_protocol",
        fake_epw_builder,
    )

    parent_folder_ph = generate_remote_data(fixture_localhost, "/remote/ph")
    builder = EpwPrepWorkChain.get_builder_from_protocol(
        codes={"ph": fixture_code("quantumespresso.ph"), "epw": epw_code},
        structure=generate_structure(),
        overrides={"do_bands_interpolation": False},
        parent_folder_ph=parent_folder_ph,
    )

    assert builder.clean_workdir.value is True
    assert builder.qpoints_distance.value == 0.5
    assert builder.kpoints_distance_scf.value == 0.15
    assert builder.kpoints_factor_nscf.value == 2
    assert builder.parent_folder_ph == parent_folder_ph
    assert "epw_bands" not in builder
    assert "projwfc" not in builder.w90_bands
    assert "open_grid" not in builder.w90_bands
    assert "structure" not in builder.w90_bands
