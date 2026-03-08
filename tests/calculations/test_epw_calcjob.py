from pathlib import Path

import pytest

from aiida import orm


def generate_kpoints_mesh(mesh):
    """Return a `KpointsData` with the provided mesh."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints_mesh(mesh)
    return kpoints


def generate_kpoints_list(points):
    """Return a `KpointsData` with the provided explicit point list."""
    kpoints = orm.KpointsData()
    kpoints.set_kpoints(points)
    return kpoints


@pytest.fixture
def generate_inputs_epw(fixture_code):
    """Return basic inputs for `EpwCalculation`."""

    def _generate_inputs(parameters=None, **overrides):
        inputs = {
            "code": fixture_code("epw.epw"),
            "parameters": orm.Dict(parameters or {"INPUTEPW": {}}),
            "kpoints": generate_kpoints_mesh([2, 2, 2]),
            "qpoints": generate_kpoints_mesh([2, 2, 1]),
            "kfpoints": generate_kpoints_mesh([4, 4, 4]),
            "qfpoints": generate_kpoints_mesh([4, 4, 2]),
            "metadata": {
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                    },
                    "max_wallclock_seconds": 1800,
                }
            },
        }
        inputs.update(overrides)
        return inputs

    return _generate_inputs


def test_epw_default_mesh_inputs(fixture_sandbox, generate_calc_job, generate_inputs_epw):
    """Test that plugin-managed meshes are written to the EPW input file."""
    calc_info = generate_calc_job(
        fixture_sandbox, "epw.epw", generate_inputs_epw()
    )

    input_contents = Path(fixture_sandbox.abspath, "aiida.in").read_text()

    assert "outdir = './out/'" in input_contents
    assert "dvscf_dir = 'save'" in input_contents
    assert "prefix = 'aiida'" in input_contents
    assert "nk1 = 2" in input_contents
    assert "nq3 = 1" in input_contents
    assert "nkf3 = 4" in input_contents
    assert "nqf2 = 4" in input_contents
    assert calc_info.retrieve_list == ["aiida.out"]


def test_epw_writes_explicit_fine_point_files(
    fixture_sandbox, generate_calc_job, generate_inputs_epw
):
    """Test that explicit fine grids are written to separate point-list files."""
    inputs = generate_inputs_epw(
        kfpoints=generate_kpoints_list([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
        qfpoints=generate_kpoints_list([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
    )

    generate_calc_job(fixture_sandbox, "epw.epw", inputs)

    input_contents = Path(fixture_sandbox.abspath, "aiida.in").read_text()
    kfpoints_contents = Path(fixture_sandbox.abspath, "kfpoints.kpt").read_text()
    qfpoints_contents = Path(fixture_sandbox.abspath, "qfpoints.kpt").read_text()

    assert "filkf = 'kfpoints.kpt'" in input_contents
    assert "filqf = 'qfpoints.kpt'" in input_contents
    assert "nkf1" not in input_contents
    assert "nqf1" not in input_contents
    assert kfpoints_contents.splitlines()[0] == "2 crystal"
    assert qfpoints_contents.splitlines()[0] == "2 crystal"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("outdir", "./custom-out"),
        ("nk1", 8),
        ("filkf", "custom.kpt"),
    ],
)
def test_epw_rejects_plugin_managed_keywords(
    fixture_sandbox, generate_calc_job, generate_inputs_epw, key, value
):
    """Test that users cannot override keywords managed by the plugin."""
    inputs = generate_inputs_epw(parameters={"INPUTEPW": {key: value}})

    with pytest.raises(ValueError, match=rf"parameters\.INPUTEPW\.{key}"):
        generate_calc_job(fixture_sandbox, "epw.epw", inputs)


def test_epw_allows_user_amass_parameter(
    fixture_sandbox, generate_calc_job, generate_inputs_epw
):
    """Test that `amass` remains user-configurable."""
    inputs = generate_inputs_epw(parameters={"INPUTEPW": {"amass": 28.085}})

    generate_calc_job(fixture_sandbox, "epw.epw", inputs)

    input_contents = Path(fixture_sandbox.abspath, "aiida.in").read_text()
    assert "amass" in input_contents


def test_epw_rejects_wannierize_with_restart_parent(
    fixture_sandbox,
    fixture_localhost,
    generate_calc_job,
    generate_inputs_epw,
    generate_remote_data,
):
    """Test that wannierization cannot be mixed with restart parents."""
    inputs = generate_inputs_epw(
        parameters={"INPUTEPW": {"wannierize": True}},
        parent_folder_chk=generate_remote_data(
            fixture_localhost, fixture_sandbox.abspath
        ),
    )

    with pytest.raises(ValueError, match="wannierize"):
        generate_calc_job(fixture_sandbox, "epw.epw", inputs)
