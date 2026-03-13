from types import SimpleNamespace

import pytest

from aiida import orm
from aiida.common import LinkType
from aiida.engine import ProcessHandlerReport
from aiida.plugins import CalculationFactory

from aiida_epw.workflows.base import EpwBaseWorkChain, validate_inputs

EpwCalculation = CalculationFactory("epw.epw")


def create_remote_data_with_creator(
    computer, remote_path, process_type, link_label="remote_folder", inputs=None
):
    """Create a `RemoteData` node with a creator that has the requested inputs."""
    creator = orm.CalcJobNode(computer=computer, process_type=process_type)
    creator.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})

    for input_label, node in (inputs or {}).items():
        if not node.is_stored:
            node.store()
        creator.base.links.add_incoming(
            node,
            link_type=LinkType.INPUT_CALC,
            link_label=input_label,
        )

    creator.store()

    remote_data = orm.RemoteData(computer=computer, remote_path=remote_path)
    remote_data.base.links.add_incoming(
        creator,
        link_type=LinkType.CREATE,
        link_label=link_label,
    )
    remote_data.store()

    return remote_data


def create_failed_epw_calculation(exit_code, remote_folder=None):
    """Return a minimal failed calculation-like object for handler tests."""
    outputs = SimpleNamespace()
    if remote_folder is not None:
        outputs.remote_folder = remote_folder

    return SimpleNamespace(
        is_failed=True,
        is_finished_ok=False,
        is_excepted=False,
        is_killed=False,
        exit_status=exit_code.status,
        exit_message=exit_code.message,
        process_label="EpwCalculation",
        pk=1,
        outputs=outputs,
    )


@pytest.fixture
def generate_inputs_epw_base(fixture_code, generate_structure):
    """Return minimal valid inputs for `EpwBaseWorkChain`."""

    def _generate_inputs_epw_base(**overrides):
        inputs = {
            "code": fixture_code("epw.epw"),
            "structure": generate_structure(),
            "parameters": orm.Dict({"INPUTEPW": {}}),
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
            "qfpoints_distance": orm.Float(0.2),
            "kfpoints_factor": orm.Int(2),
            "max_iterations": orm.Int(3),
        }
        inputs.update(overrides)
        return inputs

    return _generate_inputs_epw_base


def test_validate_inputs_requires_one_fine_grid_source():
    """Exactly one source for each fine mesh should be configured."""
    message = validate_inputs({"qfpoints_distance": orm.Float(0.2)})
    assert message == "Either `kfpoints` or `kfpoints_factor` must be specified."

    message = validate_inputs(
        {
            "qfpoints_distance": orm.Float(0.2),
            "kfpoints_factor": orm.Int(2),
        }
    )
    assert message is None


def test_validate_inputs_rejects_duplicate_fine_grid_sources(generate_kpoints_mesh):
    """Duplicated fine-grid inputs should be rejected before launch."""
    message = validate_inputs(
        {
            "qfpoints": generate_kpoints_mesh([2, 2, 2]),
            "qfpoints_distance": orm.Float(0.2),
            "kfpoints_factor": orm.Int(2),
        }
    )
    assert message == "Can only specify one of the `qfpoints`, `qfpoints_distance`."

    message = validate_inputs(
        {
            "qfpoints_distance": orm.Float(0.2),
            "kfpoints": generate_kpoints_mesh([4, 4, 4]),
            "kfpoints_factor": orm.Int(2),
        }
    )
    assert message == "Can only specify one of the `kfpoints`, `kfpoints_factor`."


def test_get_builder_from_protocol_sets_protocol_defaults(
    fixture_code, generate_structure, serialize_builder
):
    """Protocol builders should expose the EPW defaults at the workchain level."""
    builder = EpwBaseWorkChain.get_builder_from_protocol(
        code=fixture_code("epw.epw"),
        structure=generate_structure(),
        protocol="fast",
    )

    serialized = serialize_builder(builder)

    assert serialized["qfpoints_distance"] == 0.2
    assert serialized["kfpoints_factor"] == 1
    assert serialized["clean_workdir"] is True
    assert serialized["parameters"]["INPUTEPW"]["use_ws"] is True
    assert serialized["options"]["withmpi"] is True
    assert "parent_folder" not in serialized


def test_get_builder_from_protocol_accepts_parallelization_override(
    fixture_code, generate_structure
):
    """Test that protocol builders can forward the `parallelization` input."""
    builder = EpwBaseWorkChain.get_builder_from_protocol(
        code=fixture_code("epw.epw"),
        structure=generate_structure(),
        overrides={"parallelization": {"npool": 2}},
    )

    assert builder.parallelization.get_dict() == {"npool": 2}
    assert "parent_folder" not in builder


def test_setup_updates_parameters_from_wannier_parent(
    fixture_localhost,
    generate_workchain,
    generate_inputs_epw_base,
):
    """`setup` should transfer Wannier metadata into the EPW input parameters."""
    wannier_parameters = orm.Dict(
        {"mp_grid": [4, 4, 4], "num_wann": 8, "exclude_bands": [1, 4]}
    )
    parent_folder_chk = create_remote_data_with_creator(
        fixture_localhost,
        "/remote/wannier",
        "aiida.calculations:wannier90.wannier90",
        inputs={"parameters": wannier_parameters},
    )
    w90_chk_to_ukk_script = orm.RemoteData(
        computer=fixture_localhost,
        remote_path="/remote/bin/w90_chk2ukk.jl",
    ).store()

    process = generate_workchain(
        "epw.base",
        generate_inputs_epw_base(
            parent_folder_chk=parent_folder_chk,
            w90_chk_to_ukk_script=w90_chk_to_ukk_script,
        ),
    )
    process.setup()

    parameters = process.ctx.inputs.parameters.get_dict()["INPUTEPW"]
    prepend_text = process.ctx.inputs.metadata["options"]["prepend_text"]

    assert parameters["nbndsub"] == 8
    assert parameters["bands_skipped"] == "exclude_bands = 1:4"
    assert "/remote/bin/w90_chk2ukk.jl" in prepend_text
    assert "aiida.chk" in prepend_text
    assert "aiida.ukk" in prepend_text


def test_setup_updates_parameters_from_epw_restart_parent(
    fixture_localhost,
    generate_workchain,
    generate_inputs_epw_base,
    generate_remote_data,
    monkeypatch,
):
    """`setup` should reuse restart metadata from a previous EPW calculation."""
    parent_folder_epw = generate_remote_data(fixture_localhost, "/remote/epw")
    restart_calculation = SimpleNamespace(
        inputs=SimpleNamespace(
            parameters=orm.Dict(
                {
                    "INPUTEPW": {
                        "use_ws": True,
                        "nbndsub": 12,
                        "bands_skipped": "exclude_bands = 2:5",
                    }
                }
            )
        )
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.base.find_related_calculation",
        lambda _: restart_calculation,
    )

    process = generate_workchain(
        "epw.base",
        generate_inputs_epw_base(parent_folder_epw=parent_folder_epw),
    )
    process.setup()

    parameters = process.ctx.inputs.parameters.get_dict()["INPUTEPW"]

    assert parameters["use_ws"] is True
    assert parameters["nbndsub"] == 12
    assert parameters["bands_skipped"] == "exclude_bands = 2:5"


def test_validate_kpoints_uses_parent_folders(
    fixture_localhost,
    generate_kpoints_mesh,
    generate_workchain,
    generate_inputs_epw_base,
    monkeypatch,
):
    """The base workchain should derive coarse and fine meshes from its parents."""
    wannier_parameters = orm.Dict({"mp_grid": [4, 4, 4], "num_wann": 6})
    parent_folder_chk = create_remote_data_with_creator(
        fixture_localhost,
        "/remote/wannier",
        "aiida.calculations:wannier90.wannier90",
        inputs={"parameters": wannier_parameters},
    )
    parent_folder_ph = orm.RemoteData(
        computer=fixture_localhost, remote_path="/remote/ph"
    ).store()

    qpoints = generate_kpoints_mesh([2, 2, 2])
    qfpoints = generate_kpoints_mesh([3, 3, 3])

    monkeypatch.setattr(
        "aiida_epw.workflows.base.get_parent_folder_calculation",
        lambda _: SimpleNamespace(inputs=SimpleNamespace(qpoints=qpoints)),
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.base.create_kpoints_from_distance",
        lambda **_: qfpoints,
    )

    process = generate_workchain(
        "epw.base",
        generate_inputs_epw_base(
            parent_folder_chk=parent_folder_chk,
            parent_folder_ph=parent_folder_ph,
        ),
    )
    process.setup()

    assert process.validate_kpoints() is None
    assert process.ctx.inputs.kpoints.get_kpoints_mesh()[0] == [4, 4, 4]
    assert process.ctx.inputs.qpoints.get_kpoints_mesh()[0] == [2, 2, 2]
    assert process.ctx.inputs.qfpoints.get_kpoints_mesh()[0] == [3, 3, 3]
    assert process.ctx.inputs.kfpoints.get_kpoints_mesh()[0] == [6, 6, 6]


def test_validate_kpoints_uses_restart_parent_meshes(
    fixture_localhost,
    generate_kpoints_mesh,
    generate_remote_data,
    generate_workchain,
    generate_inputs_epw_base,
    monkeypatch,
):
    """Restarting from EPW should reuse the coarse meshes from the parent calculation."""
    parent_folder_epw = generate_remote_data(fixture_localhost, "/remote/epw")
    restart_calculation = SimpleNamespace(
        inputs=SimpleNamespace(
            parameters=orm.Dict({"INPUTEPW": {"use_ws": True, "nbndsub": 10}}),
            kpoints=generate_kpoints_mesh([6, 6, 6]),
            qpoints=generate_kpoints_mesh([3, 3, 3]),
        )
    )
    qfpoints = generate_kpoints_mesh([4, 4, 4])

    monkeypatch.setattr(
        "aiida_epw.workflows.base.find_related_calculation",
        lambda _: restart_calculation,
    )
    monkeypatch.setattr(
        "aiida_epw.workflows.base.create_kpoints_from_distance",
        lambda **_: qfpoints,
    )

    process = generate_workchain(
        "epw.base",
        generate_inputs_epw_base(parent_folder_epw=parent_folder_epw),
    )
    process.setup()

    assert process.validate_kpoints() is None
    assert process.ctx.inputs.kpoints.get_kpoints_mesh()[0] == [6, 6, 6]
    assert process.ctx.inputs.qpoints.get_kpoints_mesh()[0] == [3, 3, 3]
    assert process.ctx.inputs.qfpoints.get_kpoints_mesh()[0] == [4, 4, 4]
    assert process.ctx.inputs.kfpoints.get_kpoints_mesh()[0] == [8, 8, 8]


def test_handle_unrecoverable_failure(
    generate_workchain,
    generate_inputs_epw_base,
):
    """Unknown calcjob failures below 400 should abort the restart loop."""
    process = generate_workchain("epw.base", generate_inputs_epw_base())
    process.setup()

    calculation = SimpleNamespace(
        is_failed=True,
        exit_status=300,
        exit_message="synthetic failure",
        process_label="EpwCalculation",
        pk=1,
    )

    result = process.handle_unrecoverable_failure(calculation)

    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break is True
    assert result.exit_code == process.exit_codes.ERROR_UNRECOVERABLE_FAILURE


def test_handle_out_of_walltime_prepares_epw_restart(
    fixture_localhost,
    generate_remote_data,
    generate_workchain,
    generate_inputs_epw_base,
):
    """Clean EPW walltime exits should restart from the latest remote folder."""
    remote_folder = generate_remote_data(fixture_localhost, "/remote/restart")
    process = generate_workchain(
        "epw.base",
        generate_inputs_epw_base(
            parameters=orm.Dict({"INPUTEPW": {"elph": True}})
        ),
    )
    process.setup()

    calculation = create_failed_epw_calculation(
        EpwCalculation.exit_codes.ERROR_OUT_OF_WALLTIME, remote_folder=remote_folder
    )

    process.ctx.iteration = 1
    process.ctx.children = [calculation]

    result = process.inspect_process()

    assert result.status == 0

    process.prepare_process()
    parameters = process.ctx.inputs.parameters.get_dict()["INPUTEPW"]

    assert process.ctx.inputs.parent_folder_epw == remote_folder
    assert parameters["epwread"] is True
    assert getattr(process.ctx, "restart_calc", None) is None


def test_handle_out_of_walltime_enables_eliashberg_restart(
    fixture_localhost,
    generate_remote_data,
    generate_workchain,
    generate_inputs_epw_base,
):
    """Eliashberg runs should switch on `restart` after a clean walltime exit."""
    remote_folder = generate_remote_data(fixture_localhost, "/remote/eliashberg")
    process = generate_workchain(
        "epw.base",
        generate_inputs_epw_base(
            parameters=orm.Dict(
                {"INPUTEPW": {"elph": True, "eliashberg": True, "ephwrite": True}}
            )
        ),
    )
    process.setup()

    calculation = create_failed_epw_calculation(
        EpwCalculation.exit_codes.ERROR_OUT_OF_WALLTIME, remote_folder=remote_folder
    )
    process.ctx.iteration = 1
    process.ctx.children = [calculation]

    result = process.inspect_process()

    assert result.status == 0

    process.prepare_process()
    parameters = process.ctx.inputs.parameters.get_dict()["INPUTEPW"]

    assert process.ctx.inputs.parent_folder_epw == remote_folder
    assert parameters["epwread"] is True
    assert parameters["restart"] is True


def test_handle_scheduler_out_of_walltime_aborts_explicitly(
    generate_workchain,
    generate_inputs_epw_base,
):
    """Scheduler walltime exits should not be retried blindly."""
    process = generate_workchain("epw.base", generate_inputs_epw_base())
    process.setup()

    calculation = create_failed_epw_calculation(
        EpwCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME
    )

    process.ctx.iteration = 1
    process.ctx.children = [calculation]

    result = process.inspect_process()

    assert result == process.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE


def test_handle_known_unrecoverable_failure_uses_dedicated_exit_code(
    generate_workchain,
    generate_inputs_epw_base,
):
    """Known unrecoverable EPW failures should use the dedicated workchain exit code."""
    process = generate_workchain("epw.base", generate_inputs_epw_base())
    process.setup()

    calculation = create_failed_epw_calculation(
        EpwCalculation.exit_codes.ERROR_MEMORY_EXCEEDS_MAX_MEMLT
    )

    process.ctx.iteration = 1
    process.ctx.children = [calculation]

    result = process.inspect_process()

    assert result == process.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE
