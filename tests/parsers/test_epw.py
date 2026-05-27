"""Tests for the `EpwParser`."""

import textwrap

import pytest
from aiida import orm
from aiida.common import LinkType
from aiida.plugins.entry_point import (
    format_entry_point_string,
    get_entry_point_string_from_class,
)

from aiida_epw.calculations.epw import EpwCalculation
from aiida_epw.data import (
    A2fData,
    DosData,
)
from aiida_epw.parsers.epw import EpwParser


@pytest.mark.parametrize(
    "test_name",
    (
        "default",
        "isotropic_eliashberg",
        "bands",
    ),
)
def test_epw(parse_from_files, data_regression, test_name):
    """Test the `EpwParser`."""
    results, calcfunction = parse_from_files(EpwParser, test_name=test_name)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message

    data_regression_dict = {
        "output_parameters": results["output_parameters"].get_dict(),
    }
    if test_name == "default":
        assert isinstance(results["a2f"], A2fData)
        data_regression_dict["a2f"] = results["a2f"].get_array("a2f").tolist()[::50]
        data_regression_dict["lambda"] = results["a2f"].get_array("lambda").tolist()
        data_regression_dict["degaussq"] = results["a2f"].get_array("degaussq").tolist()
        assert results["a2f"].get_spectrum().shape[1] == 10
        assert results["a2f"].get_cumulative_lambda().shape[1] == 10
    if test_name == "isotropic_eliashberg":
        data_regression_dict["max_eigenvalue"] = (
            results["max_eigenvalue"].get_array("max_eigenvalue").tolist()
        )
    if test_name == "bands":
        data_regression_dict["kpoints"] = (
            results["el_band_structure"].get_kpoints().tolist()
        )
        data_regression_dict["el_band_structure"] = (
            results["el_band_structure"].get_bands().tolist()
        )
        data_regression_dict["qpoints"] = (
            results["ph_band_structure"].get_kpoints().tolist()
        )
        data_regression_dict["ph_band_structure"] = (
            results["ph_band_structure"].get_bands().tolist()
        )

    data_regression.check(data_regression_dict)


def test_epw_failed_broyden_factor(parse_from_files, data_regression):
    """Test a `epw.x` that failed due to an error in routine `mix_broyden`."""
    results, calcfunction = parse_from_files(EpwParser, "failed_broyden_factor")
    expected_exit_status = (
        EpwCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE.status
    )

    assert calcfunction.is_failed
    assert calcfunction.exit_status == expected_exit_status
    data_regression.check(
        {
            "output_parameters": results["output_parameters"].get_dict(),
        }
    )


def test_epw_reads_dos_from_output_subfolder(aiida_localhost, files_path):
    """Test that DOS data is parsed even when retrieved inside the EPW output folder."""
    parser_entry_point = get_entry_point_string_from_class(
        class_module=EpwParser.__module__, class_name=EpwParser.__name__
    )
    calc_entry_point = format_entry_point_string(
        group="aiida.calculations", name=parser_entry_point.split(":")[1]
    )

    node = orm.CalcJobNode(computer=aiida_localhost, process_type=calc_entry_point)
    node.base.attributes.set("output_filename", "aiida.out")
    node.store()

    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_tree(
        (files_path / "parsers" / "epw" / "default").as_posix()
    )
    retrieved.base.repository.put_object_from_file(
        (files_path / "tools" / "parsers" / "a2f" / "aiida.dos").as_posix(),
        "out/aiida.dos",
    )
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()

    results, calcfunction = EpwParser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "dos" in results
    assert isinstance(results["dos"], DosData)
    assert results["dos"].get_array("EDOS").shape == (160,)
    assert "IDOS" in results["dos"].get_arraynames()


def test_parse_phdos_preserves_all_smearing_columns():
    """Test that the total phonon DOS keeps every smearing series."""
    content = """w[meV] phdos[states/meV] for   3 smearing values
   0.1000000   1.0000000   2.0000000   3.0000000
   0.2000000   4.0000000   5.0000000   6.0000000
"""

    phdos = EpwParser.parse_phdos(content)

    assert phdos.get_array("Frequency").tolist() == [0.1, 0.2]
    assert phdos.get_array("PHDOS").tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]


def test_epw_calculation_registers_memory_exit_code():
    """Test that parser-side memory errors map to a defined calculation exit code."""
    assert EpwCalculation.exit_codes.ERROR_MEMORY_EXCEEDS_MAX_MEMLT.status == 313


def test_epw_calculation_registers_walltime_exit_code():
    """Test that parser-side walltime errors map to a defined calculation exit code."""
    assert EpwCalculation.exit_codes.ERROR_OUT_OF_WALLTIME.status == 400


def test_epw_out_of_walltime(aiida_localhost, tmp_path):
    """Test that internal EPW walltime errors map to `ERROR_OUT_OF_WALLTIME`."""
    parser_entry_point = get_entry_point_string_from_class(
        class_module=EpwParser.__module__, class_name=EpwParser.__name__
    )
    calc_entry_point = format_entry_point_string(
        group="aiida.calculations", name=parser_entry_point.split(":")[1]
    )
    node = orm.CalcJobNode(computer=aiida_localhost, process_type=calc_entry_point)
    node.base.attributes.set("output_filename", "aiida.out")
    node.store()

    stdout_path = tmp_path / "aiida.out"
    stdout_path.write_text(
        textwrap.dedent(
            """\
            Program EPW v.5.7 starts on 15May2023 at  3: 7:50
            Maximum CPU time exceeded
            """
        )
    )

    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_file(stdout_path.as_posix(), "aiida.out")
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()

    results, calcfunction = EpwParser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished
    assert calcfunction.is_failed
    assert (
        calcfunction.exit_status
        == EpwCalculation.exit_codes.ERROR_OUT_OF_WALLTIME.status
    )
    assert "output_parameters" in results


def test_epw_preserves_scheduler_out_of_walltime(aiida_localhost, tmp_path):
    """Test that scheduler walltime failures are not overridden by parser errors."""
    parser_entry_point = get_entry_point_string_from_class(
        class_module=EpwParser.__module__, class_name=EpwParser.__name__
    )
    calc_entry_point = format_entry_point_string(
        group="aiida.calculations", name=parser_entry_point.split(":")[1]
    )
    node = orm.CalcJobNode(computer=aiida_localhost, process_type=calc_entry_point)
    node.base.attributes.set("output_filename", "aiida.out")
    node.set_exit_status(
        EpwCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME.status
    )
    node.store()

    stdout_path = tmp_path / "aiida.out"
    stdout_path.write_text(
        textwrap.dedent(
            """\
            Program EPW v.5.7 starts on 15May2023 at  3: 7:50
            """
        )
    )

    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_file(stdout_path.as_posix(), "aiida.out")
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()

    _, calcfunction = EpwParser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished
    assert calcfunction.is_failed
    assert (
        calcfunction.exit_status
        == EpwCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME.status
    )


def test_epw_detects_scheduler_out_of_memory_from_stderr(aiida_localhost, tmp_path):
    """Scheduler OOM messages should take precedence over incomplete stdout parsing."""
    parser_entry_point = get_entry_point_string_from_class(
        class_module=EpwParser.__module__, class_name=EpwParser.__name__
    )
    calc_entry_point = format_entry_point_string(
        group="aiida.calculations", name=parser_entry_point.split(":")[1]
    )
    node = orm.CalcJobNode(computer=aiida_localhost, process_type=calc_entry_point)
    node.base.attributes.set("output_filename", "aiida.out")
    node.set_option("scheduler_stderr", "_scheduler-stderr.txt")
    node.store()

    stdout_path = tmp_path / "aiida.out"
    stdout_path.write_text(
        textwrap.dedent(
            """\
            Program EPW v.5.7 starts on 15May2023 at  3: 7:50
            """
        )
    )
    scheduler_stderr_path = tmp_path / "_scheduler-stderr.txt"
    scheduler_stderr_path.write_text(
        textwrap.dedent(
            """\
            slurmstepd: error: Detected 4 oom_kill events in StepId=13178347.0.
            srun: error: cns261: tasks 104,106: Out Of Memory
            """
        )
    )

    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_file(stdout_path.as_posix(), "aiida.out")
    retrieved.base.repository.put_object_from_file(
        scheduler_stderr_path.as_posix(), "_scheduler-stderr.txt"
    )
    retrieved.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()

    _, calcfunction = EpwParser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished
    assert calcfunction.is_failed
    assert (
        calcfunction.exit_status
        == EpwCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY.status
    )
