"""Tests for workflow helpers in ``aiida_epw.workflows.prep``."""

from pathlib import Path
from types import SimpleNamespace

from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.common.datastructures import StashMode
from aiida.engine import WorkChain
from aiida.plugins.entry_point import format_entry_point_string

from aiida_epw.workflows.prep import (
    EpwPrepWorkChain,
    get_target_basepath,
    should_run_bands_interpolation,
    validate_inputs,
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


def test_results_exposes_transformation_outputs():
    """The results step should forward the main EPW outputs."""
    retrieved = orm.FolderData()
    stash = orm.RemoteStashFolderData(
        stash_mode=StashMode.COPY,
        target_basepath="/stash/epw",
        source_list=["save"],
    )
    captured = {}
    process = SimpleNamespace(
        ctx=SimpleNamespace(
            workchain_epw=SimpleNamespace(
                outputs=SimpleNamespace(retrieved=retrieved, remote_stash=stash)
            )
        ),
        out=lambda label, value: captured.setdefault(label, value),
    )

    EpwPrepWorkChain.results(process)

    assert captured == {"retrieved": retrieved, "epw_folder": stash}


def test_inspect_wannier90_reports_child_exit_message():
    """Failed Wannier90 steps should report the underlying exit message."""
    reports = []
    process = SimpleNamespace(
        ctx=SimpleNamespace(
            w90_class_name="Wannier90BandsWorkChain",
            workchain_w90_bands=SimpleNamespace(
                pk=321,
                is_finished_ok=False,
                exit_status=401,
                exit_message="no convergence reached",
            ),
        ),
        report=reports.append,
        exit_codes=SimpleNamespace(ERROR_SUB_PROCESS_FAILED_WANNIER90="sentinel"),
    )

    result = EpwPrepWorkChain.inspect_wannier90(process)

    assert result == "sentinel"
    assert reports == [
        "Wannier90BandsWorkChain<321> failed with exit status 401: no convergence reached"
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
        EpwPrepWorkChain,
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
        EpwPrepWorkChain,
        True,
        [clean_descendant, failing_descendant, missing_remote, ignored_workchain],
        monkeypatch,
    )

    process.on_terminated()

    assert cleaned_paths == ["/remote/clean"]
    assert reports == [
        f"cleaned remote folders of calculations: {clean_descendant.pk}"
    ]
