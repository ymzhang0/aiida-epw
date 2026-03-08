"""Tests for workflow helpers and steps in ``aiida_epw.workflows.supercon``."""

from types import SimpleNamespace

import pytest

from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.engine import WorkChain
from aiida.plugins.entry_point import format_entry_point_string

from aiida_epw.workflows.base import EpwBaseWorkChain
from aiida_epw.workflows.supercon import (
    SuperConWorkChain,
    get_restart_parent_folder,
)


def create_calcjob_descendant(computer, remote_path=None):
    """Create a calcjob descendant with an optional `remote_folder` output."""
    node = orm.CalcJobNode(
        computer=computer,
        process_type=format_entry_point_string("aiida.calculations", "epw.epw"),
    )
    node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    node.store()

    if remote_path is not None:
        remote_folder = orm.RemoteData(computer=computer, remote_path=remote_path)
        remote_folder.base.links.add_incoming(
            node,
            link_type=LinkType.CREATE,
            link_label="remote_folder",
        )
        remote_folder.store()

    return node


def create_workchain_node_with_outputs(output_values):
    """Create an `epw.base` workchain node exposing the given outputs."""
    node = orm.WorkChainNode(
        process_type=format_entry_point_string("aiida.workflows", "epw.base")
    )
    node.store()

    for label, value in output_values.items():
        if not value.is_stored:
            value.store()
        value.base.links.add_incoming(
            node,
            link_type=LinkType.RETURN,
            link_label=label,
        )

    return node


def make_cleanup_process(workchain_cls, clean_workdir, descendants, monkeypatch):
    """Create a lightweight workchain instance for testing `on_terminated`."""
    monkeypatch.setattr(WorkChain, "on_terminated", lambda self: None)
    monkeypatch.setattr(
        workchain_cls,
        "inputs",
        property(lambda self: self._inputs),
        raising=False,
    )

    process = object.__new__(workchain_cls)
    process._inputs = AttributeDict({"clean_workdir": orm.Bool(clean_workdir)})
    process._node = SimpleNamespace(called_descendants=descendants)
    reports = []
    process.report = reports.append

    return process, reports


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


def test_results_exposes_interpolation_outputs_without_final_steps(
    generate_workchain, generate_inputs_supercon
):
    """Without final runs, the workchain should still expose the converged interpolation outputs."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()
    process.ctx.epw_interp = [
        create_workchain_node_with_outputs(
            {"output_parameters": orm.Dict({"Allen_Dynes_Tc": 10.0})}
        )
    ]

    process.results()
    process.update_outputs()

    assert process.node.outputs.epw_final_a2f.output_parameters.get_dict() == {
        "Allen_Dynes_Tc": 10.0
    }
    assert sorted(process.node.base.links.get_outgoing().all_link_labels()) == [
        "epw_final_a2f__output_parameters"
    ]


def test_results_exposes_final_iso_and_aniso_outputs(
    generate_workchain, generate_inputs_supercon
):
    """Final isotropic and anisotropic outputs should be exposed under their namespaces."""
    process = generate_workchain("epw.supercon", generate_inputs_supercon())
    process.setup()
    process.ctx.epw_interp = [
        create_workchain_node_with_outputs(
            {"output_parameters": orm.Dict({"Allen_Dynes_Tc": 10.0})}
        )
    ]
    process.ctx.final_epw_iso = create_workchain_node_with_outputs(
        {"output_parameters": orm.Dict({"Tc": 9.5})}
    )
    process.ctx.final_epw_aniso = create_workchain_node_with_outputs(
        {"output_parameters": orm.Dict({"Tc": 9.2})}
    )

    process.results()
    process.update_outputs()

    assert process.node.outputs.epw_final_iso.output_parameters.get_dict() == {
        "Tc": 9.5
    }
    assert process.node.outputs.epw_final_aniso.output_parameters.get_dict() == {
        "Tc": 9.2
    }
    assert sorted(process.node.base.links.get_outgoing().all_link_labels()) == [
        "epw_final_a2f__output_parameters",
        "epw_final_aniso__output_parameters",
        "epw_final_iso__output_parameters",
    ]


def test_on_terminated_skips_cleanup_when_disabled(
    fixture_localhost,
    monkeypatch,
):
    """Disabling cleanup should leave descendant folders untouched."""
    cleaned_paths = []
    descendant = create_calcjob_descendant(fixture_localhost, "/remote/keep")

    monkeypatch.setattr(
        orm.RemoteData,
        "_clean",
        lambda self: cleaned_paths.append(self.get_remote_path()),
    )

    process, reports = make_cleanup_process(
        SuperConWorkChain,
        False,
        [descendant],
        monkeypatch,
    )

    process.on_terminated()

    assert cleaned_paths == []
    assert reports == ["remote folders will not be cleaned"]


def test_on_terminated_cleans_calcjob_remote_folders(
    fixture_localhost,
    monkeypatch,
):
    """Cleanup should touch only calcjob descendants with removable remote folders."""
    cleaned_paths = []
    clean_descendant = create_calcjob_descendant(fixture_localhost, "/remote/clean")
    failing_descendant = create_calcjob_descendant(fixture_localhost, "/remote/fail")
    missing_remote = create_calcjob_descendant(fixture_localhost)
    ignored_workchain = orm.WorkChainNode(
        process_type=format_entry_point_string("aiida.workflows", "epw.base")
    )
    ignored_workchain.store()

    def fake_clean(self):
        if self.get_remote_path() == "/remote/fail":
            raise OSError("synthetic failure")
        cleaned_paths.append(self.get_remote_path())

    monkeypatch.setattr(orm.RemoteData, "_clean", fake_clean)

    process, reports = make_cleanup_process(
        SuperConWorkChain,
        True,
        [clean_descendant, failing_descendant, missing_remote, ignored_workchain],
        monkeypatch,
    )

    process.on_terminated()

    assert cleaned_paths == ["/remote/clean"]
    assert reports == [
        f"cleaned remote folders of calculations: {clean_descendant.pk}"
    ]
