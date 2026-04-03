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
    """The preparation workflow requires either Wannier90 or direct EPW Wannierization."""
    message = validate_inputs({"ph_base": {}, "epw_base": {}, "epw_bands": {}})

    assert (
        message
        == "Either provide `w90_bands` inputs or set `epw_base.parameters.INPUTEPW.wannierize = True`."
    )


def test_validate_inputs_rejects_mutually_exclusive_w90_and_epw_wannierize():
    """The prep workflow should not accept both Wannierization entry points at once."""
    message = validate_inputs(
        {
            "w90_bands": {},
            "scf": {},
            "nscf": {},
            "ph_base": {},
            "epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}},
        }
    )

    assert (
        message
        == "`w90_bands` inputs and `epw_base.parameters.INPUTEPW.wannierize = True` are mutually exclusive."
    )


def test_validate_inputs_requires_scf_and_nscf_for_epw_wannierize():
    """Direct EPW Wannierization should require standalone PW inputs."""
    message = validate_inputs(
        {
            "ph_base": {},
            "epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}},
        }
    )

    assert (
        message
        == "`scf` and `nscf` inputs are required when `epw_base.parameters.INPUTEPW.wannierize = True`."
    )


def test_validate_inputs_allows_epw_bands_for_epw_wannierize():
    """Direct EPW Wannierization can still request the EPW bands branch."""
    message = validate_inputs(
        {
            "scf": {},
            "nscf": {},
            "ph_base": {},
            "epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}},
            "epw_bands": {},
        }
    )

    assert message is None


def test_validate_inputs_allows_skipping_band_interpolation_namespace():
    """The bands namespace remains optional."""
    message = validate_inputs(
        {
            "w90_bands": {},
            "ph_base": {},
            "epw_base": {},
        }
    )

    assert message is None


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
    assert not EpwPrepWorkChain.should_run_wannier90(
        SimpleNamespace(
            inputs={
                "w90_bands": {},
                "epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}},
            }
        )
    )
    assert not EpwPrepWorkChain.should_run_wannier90(SimpleNamespace(inputs={}))


def test_should_run_scf_and_nscf_follow_epw_wannierize():
    """The standalone PW branches should only run for direct EPW Wannierization."""
    process = SimpleNamespace(
        inputs={"epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}}}
    )

    assert EpwPrepWorkChain.should_run_scf(process)
    assert EpwPrepWorkChain.should_run_nscf(process)


def test_should_run_epw_bands_checks_namespace_presence():
    """The workchain should run the bands branch whenever the namespace is present."""
    assert EpwPrepWorkChain.should_run_epw_bands(SimpleNamespace(inputs={"epw_bands": {}}))
    assert not EpwPrepWorkChain.should_run_epw_bands(SimpleNamespace(inputs={}))


def test_generate_reciprocal_points_splits_scf_and_nscf_meshes_for_direct_wannierize(
    generate_structure,
    monkeypatch,
):
    """Direct EPW Wannierization should keep the SCF and NSCF coarse meshes separate."""
    def fake_create_kpoints_from_distance(**inputs):
        kpoints = orm.KpointsData()
        call_link_label = inputs["metadata"]["call_link_label"]
        if call_link_label == "create_qpoints_from_distance":
            kpoints.set_kpoints_mesh([3, 3, 3])
        else:
            kpoints.set_kpoints_mesh([4, 4, 4])
        return kpoints

    monkeypatch.setattr(
        "aiida_epw.workflows.prep.create_kpoints_from_distance",
        fake_create_kpoints_from_distance,
    )

    process = SimpleNamespace(
        inputs=AttributeDict(
            {
                "structure": generate_structure(),
                "qpoints_distance": orm.Float(0.3),
                "kpoints_distance_scf": orm.Float(0.15),
                "kpoints_factor_nscf": orm.Int(2),
                "epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}},
            }
        ),
        ctx=AttributeDict(),
    )

    EpwPrepWorkChain.generate_reciprocal_points(process)

    assert process.ctx.qpoints.get_kpoints_mesh()[0] == [3, 3, 3]
    assert process.ctx.kpoints_scf.get_kpoints_mesh()[0] == [4, 4, 4]
    assert process.ctx.kpoints_nscf.get_kpoints_mesh()[0] == [6, 6, 6]


def test_generate_reciprocal_points_prefers_restart_qpoints_from_parent_ph(
    generate_structure,
    fixture_localhost,
    generate_remote_data,
    monkeypatch,
):
    """Restarting from an existing PhCalculation should make its q-points authoritative."""
    restart_qpoints = orm.KpointsData()
    restart_qpoints.set_kpoints_mesh([5, 5, 5])
    restart_parent = generate_remote_data(fixture_localhost, "/remote/ph-restart")

    def fake_create_kpoints_from_distance(**inputs):
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh([4, 4, 4])
        return kpoints

    monkeypatch.setattr(
        "aiida_epw.workflows.prep.create_kpoints_from_distance",
        fake_create_kpoints_from_distance,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.get_parent_folder_calculation",
        lambda _folder: SimpleNamespace(
            process_label="PhCalculation",
            inputs=SimpleNamespace(qpoints=restart_qpoints),
        ),
    )

    process = SimpleNamespace(
        inputs=AttributeDict(
            {
                "structure": generate_structure(),
                "parent_folder_ph": restart_parent,
                "qpoints_distance": orm.Float(0.3),
                "kpoints_distance_scf": orm.Float(0.15),
                "kpoints_factor_nscf": orm.Int(2),
                "epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}},
            }
        ),
        ctx=AttributeDict(),
    )

    EpwPrepWorkChain.generate_reciprocal_points(process)

    assert process.ctx.qpoints == restart_qpoints
    assert process.ctx.kpoints_nscf.get_kpoints_mesh()[0] == [10, 10, 10]


def test_get_builder_from_protocol_builds_epw_bands_by_default(
    fixture_code,
    fixture_localhost,
    generate_remote_data,
    generate_structure,
    monkeypatch,
):
    """The prep protocol builder should keep the EPW bands namespace from the protocol."""
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
        parent_folder_ph=parent_folder_ph,
    )

    assert builder.clean_workdir.value is True
    assert builder.qpoints_distance.value == 0.5
    assert builder.kpoints_distance_scf.value == 0.15
    assert builder.kpoints_factor_nscf.value == 2
    assert builder.parent_folder_ph == parent_folder_ph
    assert "epw_bands" in builder
    assert "projwfc" not in builder.w90_bands
    assert "open_grid" not in builder.w90_bands
    assert "structure" not in builder.w90_bands


def test_get_builder_from_protocol_uses_standalone_pw_for_epw_wannierize(
    fixture_code,
    generate_structure,
    monkeypatch,
):
    """Direct EPW Wannierization should build standalone SCF/NSCF namespaces instead of `w90_bands`."""
    from aiida.common import AttributeDict

    epw_code = fixture_code("epw.epw")
    captured = {"pw_overrides": []}

    def fake_pw_builder(*args, **kwargs):
        captured["pw_overrides"].append(kwargs["overrides"])
        builder = AttributeDict()
        builder.pw = AttributeDict()
        builder.clean_workdir = orm.Bool(False)
        builder.kpoints_distance = orm.Float(0.2)
        return builder

    def fake_ph_builder(*args, **kwargs):
        builder = AttributeDict()
        builder.clean_workdir = orm.Bool(False)
        builder.qpoints_distance = orm.Float(0.4)
        return builder

    def fake_epw_builder(*args, **kwargs):
        builder = AttributeDict()
        builder.code = epw_code
        builder.parameters = orm.Dict({"INPUTEPW": {"wannierize": True}})
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
        "aiida_epw.workflows.prep.PwBaseWorkChain.get_builder_from_protocol",
        fake_pw_builder,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.PhBaseWorkChain.get_builder_from_protocol",
        fake_ph_builder,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.prep.EpwBaseWorkChain.get_builder_from_protocol",
        fake_epw_builder,
    )

    builder = EpwPrepWorkChain.get_builder_from_protocol(
        codes={
            "pw": fixture_code("quantumespresso.pw"),
            "ph": fixture_code("quantumespresso.ph"),
            "epw": epw_code,
        },
        structure=generate_structure(),
        overrides={"epw_base": {"parameters": {"INPUTEPW": {"wannierize": True}}}},
    )

    assert "w90_bands" not in builder
    assert "scf" in builder
    assert "nscf" in builder
    assert "epw_bands" in builder
    assert captured["pw_overrides"][0]["pseudo_family"] == "PseudoDojo/0.5/PBE/SR/standard/upf"
    assert captured["pw_overrides"][1]["pseudo_family"] == "PseudoDojo/0.5/PBE/SR/standard/upf"


def test_run_epw_skips_chk_parent_for_direct_wannierize(
    fixture_localhost,
    generate_remote_data,
):
    """Direct EPW Wannierization should not pass a Wannier90 checkpoint parent."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh([5, 5, 5])
    qpoints = orm.KpointsData()
    qpoints.set_kpoints_mesh([5, 5, 5])

    captured = {}

    def submit(_process_class, **inputs):
        captured["inputs"] = inputs
        return SimpleNamespace(pk=321)

    process = SimpleNamespace(
        inputs=AttributeDict({"structure": orm.StructureData()}),
        ctx=AttributeDict(
            {
                "parent_folder_ph": generate_remote_data(
                    fixture_localhost, "/remote/ph"
                ),
                "workchain_ph": SimpleNamespace(
                    outputs=SimpleNamespace(
                        remote_folder=generate_remote_data(
                            fixture_localhost, "/remote/ph"
                        )
                    )
                ),
                "parent_folder_nscf": generate_remote_data(
                    fixture_localhost, "/remote/nscf"
                ),
                "kpoints_nscf": kpoints,
                "qpoints": qpoints,
            }
        ),
        exposed_inputs=lambda *_args, **_kwargs: AttributeDict(
            {
                "metadata": AttributeDict(),
                "parameters": orm.Dict({"INPUTEPW": {"wannierize": True}}),
                "options": orm.Dict(
                    {
                        "resources": {
                            "num_machines": 1,
                            "num_mpiprocs_per_machine": 1,
                        },
                        "max_wallclock_seconds": 1800,
                        "withmpi": True,
                    }
                ),
            }
        ),
        submit=submit,
        report=lambda *_args, **_kwargs: None,
    )

    EpwPrepWorkChain.run_epw(process)

    assert "parent_folder_chk" not in captured["inputs"]
    assert captured["inputs"]["parent_folder_nscf"] == process.ctx.parent_folder_nscf


def test_run_ph_prefers_restart_qpoints_from_parent_folder_ph(
    fixture_localhost,
    generate_remote_data,
    monkeypatch,
):
    """Restarting from an existing PhCalculation should reuse its q-points."""
    restart_qpoints = orm.KpointsData()
    restart_qpoints.set_kpoints_mesh([3, 3, 3])
    generated_qpoints = orm.KpointsData()
    generated_qpoints.set_kpoints_mesh([5, 5, 5])
    restart_parent = generate_remote_data(fixture_localhost, "/remote/ph-restart")

    captured = {}

    def submit(_process_class, **inputs):
        captured["inputs"] = inputs
        return SimpleNamespace(pk=654)

    monkeypatch.setattr(
        "aiida_epw.workflows.prep.get_parent_folder_calculation",
        lambda _folder: SimpleNamespace(
            process_label="PhCalculation",
            inputs=SimpleNamespace(qpoints=restart_qpoints),
        ),
    )

    process = SimpleNamespace(
        inputs=AttributeDict({"parent_folder_ph": restart_parent}),
        ctx=AttributeDict(
            {
                "qpoints": generated_qpoints,
                "parent_folder_scf": generate_remote_data(
                    fixture_localhost, "/remote/scf"
                ),
            }
        ),
        exposed_inputs=lambda *_args, **_kwargs: AttributeDict(
            {
                "ph": AttributeDict(),
                "metadata": AttributeDict(),
            }
        ),
        submit=submit,
        report=lambda *_args, **_kwargs: None,
    )

    EpwPrepWorkChain.run_ph(process)

    assert captured["inputs"]["ph"]["parent_folder"] == restart_parent
    assert captured["inputs"]["ph"]["qpoints"] == restart_qpoints
    assert captured["inputs"]["qpoints"] == generated_qpoints


def test_get_builder_from_protocol_backfills_stash_mode_for_current_aiida(
    fixture_code,
    generate_structure,
    monkeypatch,
):
    """The prep builder should normalize stash options for current aiida-core validation."""
    from aiida.common import AttributeDict

    captured = {}
    epw_code = fixture_code("epw.epw")
    ph_code = fixture_code("quantumespresso.ph")

    def fake_w90_builder(*args, **kwargs):
        builder = AttributeDict()
        builder.structure = generate_structure()
        builder.open_grid = {}
        builder.projwfc = {}
        return builder

    def fake_ph_builder(*args, **kwargs):
        captured["ph_overrides"] = kwargs["overrides"]
        builder = AttributeDict()
        builder.clean_workdir = orm.Bool(False)
        builder.qpoints_distance = orm.Float(0.4)
        return builder

    def fake_epw_builder(*args, **kwargs):
        captured.setdefault("epw_overrides", []).append(kwargs["overrides"])
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

    EpwPrepWorkChain.get_builder_from_protocol(
        codes={"ph": ph_code, "epw": epw_code},
        structure=generate_structure(),
        protocol="fast",
    )

    assert captured["ph_overrides"]["ph"]["metadata"]["options"]["stash"]["stash_mode"] == StashMode.COPY.value
    assert "target_base" in captured["ph_overrides"]["ph"]["metadata"]["options"]["stash"]
    assert captured["epw_overrides"][0]["options"]["stash"]["stash_mode"] == StashMode.COPY.value
    assert "target_base" in captured["epw_overrides"][0]["options"]["stash"]


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
