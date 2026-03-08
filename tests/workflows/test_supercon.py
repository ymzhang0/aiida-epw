"""Tests for workflow helpers and steps in ``aiida_epw.workflows.supercon``."""

from types import SimpleNamespace

import pytest

from aiida import orm

from aiida_epw.workflows.base import EpwBaseWorkChain
from aiida_epw.workflows.supercon import (
    SuperConWorkChain,
    get_restart_parent_folder,
)


def make_process_labelled_outputs(process_label, **outputs):
    """Create a lightweight object that mimics the outputs interface of a process node."""
    return SimpleNamespace(process_label=process_label, outputs=SimpleNamespace(**outputs))


@pytest.fixture
def generate_inputs_supercon(
    fixture_code, fixture_localhost, generate_kpoints_mesh, generate_remote_data, generate_structure
):
    """Return minimal valid inputs for `SuperConWorkChain` step tests."""

    def _generate_inputs_supercon(**overrides):
        options = orm.Dict(
            {
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                },
                "max_wallclock_seconds": 1800,
                "withmpi": True,
            }
        )
        fine_mesh = generate_kpoints_mesh([4, 4, 4])

        inputs = {
            "structure": generate_structure(),
            "parent_folder_epw": generate_remote_data(fixture_localhost, "/remote/epw"),
            "interpolation_distance": orm.List(list=[0.1, 0.2]),
            "kfpoints_factor": orm.Int(2),
            "convergence_threshold": orm.Float(0.1),
            "always_run_final": orm.Bool(False),
            "clean_workdir": orm.Bool(False),
            "epw_interp": {
                "code": fixture_code("epw.epw"),
                "parameters": orm.Dict({"INPUTEPW": {"eliashberg": True}}),
                "options": options,
                "qfpoints_distance": orm.Float(0.1),
                "kfpoints_factor": orm.Int(2),
                "max_iterations": orm.Int(2),
            },
            "epw_final_iso": {
                "code": fixture_code("epw.epw"),
                "parameters": orm.Dict({"INPUTEPW": {"liso": True, "eliashberg": True}}),
                "options": options,
                "qfpoints": fine_mesh,
                "kfpoints": fine_mesh,
                "max_iterations": orm.Int(2),
            },
            "epw_final_aniso": {
                "code": fixture_code("epw.epw"),
                "parameters": orm.Dict({"INPUTEPW": {"laniso": True, "eliashberg": True}}),
                "options": options,
                "qfpoints": fine_mesh,
                "kfpoints": fine_mesh,
                "max_iterations": orm.Int(2),
            },
        }
        inputs.update(overrides)
        return inputs

    return _generate_inputs_supercon


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


def test_get_builder_from_protocol_uses_parent_epw_inputs(
    fixture_code,
    fixture_localhost,
    generate_kpoints_mesh,
    generate_remote_data,
    generate_structure,
    serialize_builder,
):
    """Protocol builders should inherit the structure and coarse meshes from the parent EPW run."""
    parent_folder_epw = generate_remote_data(fixture_localhost, "/remote/restart")
    parent_epw = SimpleNamespace(
        process_label="EpwBaseWorkChain",
        inputs=SimpleNamespace(
            structure=generate_structure(),
            kpoints=generate_kpoints_mesh([6, 6, 6]),
            qpoints=generate_kpoints_mesh([3, 3, 3]),
        ),
    )

    builder = SuperConWorkChain.get_builder_from_protocol(
        epw_code=fixture_code("epw.epw"),
        parent_epw=parent_epw,
        parent_folder_epw=parent_folder_epw,
        protocol="fast",
    )
    serialized = serialize_builder(builder)

    assert builder.structure == parent_epw.inputs.structure
    assert builder.parent_folder_epw == parent_folder_epw
    assert builder.interpolation_distance.value == 0.5
    assert serialized["epw_interp"]["kpoints"] == ([6, 6, 6], [0.0, 0.0, 0.0])
    assert serialized["epw_interp"]["qpoints"] == ([3, 3, 3], [0.0, 0.0, 0.0])


def test_get_builder_from_protocol_accepts_prep_parent(
    fixture_code,
    fixture_localhost,
    generate_kpoints_mesh,
    generate_remote_data,
    generate_structure,
    serialize_builder,
):
    """The supercon builder should also accept `EpwPrepWorkChain` parents."""
    structure = generate_structure()
    parent_folder_epw = generate_remote_data(fixture_localhost, "/remote/restart")
    epw_source = SimpleNamespace(
        inputs=SimpleNamespace(
            structure=structure,
            kpoints=generate_kpoints_mesh([8, 8, 8]),
            qpoints=generate_kpoints_mesh([4, 4, 4]),
        )
    )
    parent_epw = SimpleNamespace(
        process_label="EpwPrepWorkChain",
        inputs=SimpleNamespace(structure=structure),
        outputs=SimpleNamespace(epw_folder=parent_folder_epw),
        base=SimpleNamespace(
            links=SimpleNamespace(
                get_outgoing=lambda **_: SimpleNamespace(
                    first=lambda: SimpleNamespace(node=epw_source)
                )
            )
        ),
    )

    builder = SuperConWorkChain.get_builder_from_protocol(
        epw_code=fixture_code("epw.epw"),
        parent_epw=parent_epw,
        protocol="fast",
    )
    serialized = serialize_builder(builder)

    assert builder.parent_folder_epw == parent_folder_epw
    assert serialized["epw_interp"]["kpoints"] == ([8, 8, 8], [0.0, 0.0, 0.0])
    assert serialized["epw_interp"]["qpoints"] == ([4, 4, 4], [0.0, 0.0, 0.0])


def test_setup_sorts_interpolation_distances(generate_workchain, generate_inputs_supercon):
    """Interpolation distances should be normalized into an ascending list."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()

    assert [value.value for value in process.ctx.interpolation_list] == [0.1, 0.2]
    assert process.ctx.iteration == 0
    assert process.ctx.is_converged is False


def test_should_run_conv_marks_convergence(generate_workchain, generate_inputs_supercon):
    """Three Allen-Dynes values should be enough to stop the convergence loop."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()
    process.ctx.epw_interp = [
        SimpleNamespace(outputs=SimpleNamespace(output_parameters={"Allen_Dynes_Tc": 10.0})),
        SimpleNamespace(outputs=SimpleNamespace(output_parameters={"Allen_Dynes_Tc": 10.5})),
        SimpleNamespace(outputs=SimpleNamespace(output_parameters={"Allen_Dynes_Tc": 10.0})),
    ]

    assert process.should_run_conv() is False
    assert process.ctx.is_converged.value is True


def test_should_run_final_returns_exit_code_when_not_converged(
    generate_workchain, generate_inputs_supercon
):
    """The final steps should abort cleanly when convergence was not reached."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()

    result = process.should_run_final()

    assert result == process.exit_codes.ERROR_ALLEN_DYNES_NOT_CONVERGED


def test_run_conv_passes_restart_inputs(
    generate_workchain,
    generate_inputs_supercon,
):
    """The convergence run should pass restart metadata and update `degaussq`."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()
    process.ctx.degaussq = 0.03
    captured = {}

    def fake_submit(process_class, **inputs):
        captured["process_class"] = process_class
        captured["inputs"] = inputs
        node = orm.WorkChainNode(process_type="aiida.workflows:epw.base")
        node.store()
        return node

    process.submit = fake_submit

    process.run_conv()

    submitted_inputs = captured["inputs"]
    assert captured["process_class"] is EpwBaseWorkChain
    assert process.ctx.iteration == 1
    assert submitted_inputs["structure"] == process.inputs.structure
    assert submitted_inputs["parent_folder_epw"] == process.inputs.parent_folder_epw
    assert submitted_inputs["metadata"]["call_link_label"] == "conv_01"
    assert submitted_inputs["qfpoints_distance"].value == 0.2
    assert submitted_inputs["parameters"].get_dict()["INPUTEPW"]["degaussq"] == 0.03


def test_run_final_epw_iso_reuses_restart_meshes(
    generate_kpoints_mesh,
    generate_workchain,
    generate_inputs_supercon,
    monkeypatch,
):
    """The final isotropic run should reuse restart folders and fine meshes."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()
    process.ctx.degaussq = 0.04
    process.ctx.epw_interp = [SimpleNamespace()]

    restart_parent = process.inputs.parent_folder_epw
    restart_calculation = SimpleNamespace(
        inputs=SimpleNamespace(
            kfpoints=generate_kpoints_mesh([8, 8, 8]),
            qfpoints=generate_kpoints_mesh([4, 4, 4]),
        )
    )
    captured = {}

    monkeypatch.setattr(
        "aiida_epw.workflows.supercon.get_restart_parent_folder",
        lambda _: restart_parent,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.supercon.find_related_calculation",
        lambda _: restart_calculation,
    )

    def fake_submit(process_class, **inputs):
        captured["process_class"] = process_class
        captured["inputs"] = inputs
        return SimpleNamespace(pk=654)

    process.submit = fake_submit

    process.run_final_epw_iso()

    submitted_inputs = captured["inputs"]
    assert submitted_inputs["parent_folder_epw"] == restart_parent
    assert submitted_inputs["kfpoints"].get_kpoints_mesh()[0] == [8, 8, 8]
    assert submitted_inputs["qfpoints"].get_kpoints_mesh()[0] == [4, 4, 4]
    assert submitted_inputs["metadata"]["call_link_label"] == "epw_final_iso"
    assert submitted_inputs["parameters"].get_dict()["INPUTEPW"]["degaussq"] == 0.04
