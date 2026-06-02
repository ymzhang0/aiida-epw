"""Tests for helper parsing functions in ``aiida_epw.tools.parsers``.

These tests deliberately use real EPW output files placed under ``tests/files``
instead of synthetic strings, and rely on ``pytest-regressions`` to freeze
their parsed output.
"""

from pathlib import Path

import numpy
import pytest

from aiida_epw.tools import parsers


def test_parse_epw_bands(files_path: Path, data_regression):
    """Parse an existing ``band.eig`` file and regress on the main arrays."""
    bands_path = files_path / "tools" / "parsers" / "bands" / "band.eig"
    content = bands_path.read_text()

    parsed = parsers.parse_epw_bands(content)

    # Basic sanity checks
    assert "kpoints" in parsed
    assert "bands" in parsed
    assert isinstance(parsed["kpoints"], numpy.ndarray)
    assert isinstance(parsed["bands"], numpy.ndarray)
    assert parsed["kpoints"].shape[0] == parsed["bands"].shape[0]

    regression_data = {
        "kpoints": parsed["kpoints"].tolist(),
        "bands": parsed["bands"].tolist(),
    }
    data_regression.check(regression_data)


def test_parse_epw_a2f(files_path: Path, data_regression):
    """Parse an existing ``aiida.a2f`` file and regress on arrays and metadata."""
    a2f_path = files_path / "tools" / "parsers" / "a2f" / "aiida.a2f"
    content = a2f_path.read_text()

    parsed = parsers.parse_epw_a2f(content)

    for key in ("frequency", "a2f", "lambda", "phonon_smearing"):
        assert key in parsed
        assert isinstance(parsed[key], numpy.ndarray)

    assert parsed["frequency"].ndim == 1
    assert parsed["a2f"].ndim == 2
    assert parsed["frequency"].shape[0] == parsed["a2f"].shape[0]

    regression_data = {
        "frequency": parsed["frequency"].tolist()[:10],
        "a2f": parsed["a2f"].tolist()[:10],
        "lambda": parsed["lambda"].tolist(),
        "phonon_smearing": parsed["phonon_smearing"].tolist(),
        "electron_smearing": float(parsed["electron_smearing"]),
        "fermi_window": float(parsed["fermi_window"]),
        "summed_elph_coupling": float(parsed["summed_elph_coupling"]),
    }
    data_regression.check(regression_data)


def test_parse_epw_max_eigenvalue(files_path: Path, data_regression):
    """Parse the max eigenvalue table from a real ``aiida.out`` file."""
    stdout_path = (
        files_path / "tools" / "parsers" / "linearized_iso_eliashberg" / "aiida.out"
    )
    content = stdout_path.read_text()

    parsed = parsers.parse_epw_max_eigenvalue(content)

    assert "max_eigenvalue" in parsed
    assert isinstance(parsed["max_eigenvalue"], numpy.ndarray)

    regression_data = {
        "max_eigenvalue": parsed["max_eigenvalue"].tolist(),
    }
    data_regression.check(regression_data)


def test_parse_epw_dos(files_path: Path, data_regression):
    """Parse an existing ``aiida.dos`` file and regress on the DOS arrays."""
    dos_path = files_path / "tools" / "parsers" / "a2f" / "aiida.dos"
    content = dos_path.read_text()

    parsed = parsers.parse_epw_dos(content)

    assert set(parsed.keys()) == {"energy", "dos", "integrated_dos"}

    regression_data = {
        "energy": parsed["energy"].tolist()[:10],
        "dos": parsed["dos"].tolist()[:10],
        "integrated_dos": parsed["integrated_dos"].tolist()[:10],
    }
    data_regression.check(regression_data)


def test_parse_epw_phdos(files_path: Path, data_regression):
    """Parse an existing ``aiida.phdos`` file and regress on the PHDOS arrays."""
    phdos_path = files_path / "tools" / "parsers" / "a2f" / "aiida.phdos"
    content = phdos_path.read_text()

    parsed = parsers.parse_epw_phdos(content)

    assert set(parsed.keys()) == {"frequency", "phdos", "num_smearings"}

    regression_data = {
        "num_smearings": parsed["num_smearings"],
        "frequency": parsed["frequency"].tolist()[:10],
        "phdos": parsed["phdos"].tolist()[:10],
    }
    data_regression.check(regression_data)


def test_parse_epw_a2f_proj(files_path: Path):
    """Parse an existing ``aiida.a2f_proj`` file and validate the projected spectrum payload."""
    a2f_proj_path = files_path / "tools" / "parsers" / "a2f" / "aiida.a2f_proj"
    content = a2f_proj_path.read_text()

    parsed = parsers.parse_epw_a2f_proj(content)

    assert set(parsed) == {
        "frequency",
        "a2f",
        "projected_a2f",
        "lambda_int",
        "lambda_sum",
    }
    assert parsed["frequency"].shape == (500,)
    assert parsed["a2f"].shape == (500,)
    assert parsed["projected_a2f"].shape == (500, 3)
    assert parsed["lambda_int"] == pytest.approx(1.9917789)
    assert parsed["lambda_sum"] == pytest.approx(1.9853134)


def test_parse_epw_phdos_proj(files_path: Path):
    """Parse an existing ``aiida.phdos_proj`` file and validate the projected spectrum payload."""
    phdos_proj_path = files_path / "tools" / "parsers" / "a2f" / "aiida.phdos_proj"
    content = phdos_proj_path.read_text()

    parsed = parsers.parse_epw_phdos_proj(content)

    assert set(parsed) == {
        "frequency",
        "phdos",
        "projected_phdos",
    }
    assert parsed["frequency"].shape == (500,)
    assert parsed["phdos"].shape == (500,)
    assert parsed["projected_phdos"].shape == (500, 3)


def test_parse_epw_lambda_fs():
    """Parse a synthetic ``lambda_FS`` table."""
    content = """# kx ky kz band Enk lambda
 0.0000 0.1000 0.2000 1 0.3000 0.9000
 0.5000 0.6000 0.7000 2 0.4000 1.1000
"""

    parsed = parsers.parse_epw_lambda_fs(content)

    assert parsed["kpoints"].tolist() == [[0.0, 0.1, 0.2], [0.5, 0.6, 0.7]]
    assert parsed["band"].tolist() == [1.0, 2.0]
    assert parsed["energy"].tolist() == [0.3, 0.4]
    assert parsed["lambda"].tolist() == [0.9, 1.1]
    assert parsed["energy_units"] == "eV"


def test_parse_epw_lambda_k_pairs():
    """Parse a synthetic ``lambda_k_pairs`` table."""
    content = """# lambda_nk rho
 0.1000 1.5000
 0.2000 2.5000
"""

    parsed = parsers.parse_epw_lambda_k_pairs(content)

    assert parsed["energy"].tolist() == [0.1, 0.2]
    assert parsed["dos"].tolist() == [1.5, 2.5]
    assert parsed["integrated_dos"] is None


def test_parse_epw_lambda_pairs(files_path: Path):
    """Parse a real ``lambda_pairs`` file as a DOS-like distribution."""
    lambda_pairs_path = (
        files_path / "tools" / "parsers" / "fbw_aniso_eliashberg" / "MgB2.lambda_pairs"
    )
    content = lambda_pairs_path.read_text()

    parsed = parsers.parse_epw_lambda_pairs(content)

    assert parsed["energy"].shape == (599,)
    assert parsed["dos"].shape == (599,)
    assert parsed["integrated_dos"] is None
    assert parsed["energy"][0] == pytest.approx(0.0)
    assert parsed["dos"][0] == pytest.approx(0.029540468306)


def test_parse_epw_imag_iso(files_path: Path, data_regression):
    """Parse isotropic ``imag_iso`` files from a folder mapping."""
    iso_dir = files_path / "tools" / "parsers" / "full_iso_eliashberg"

    file_contents = {
        path.name: path.read_text()
        for path in sorted(iso_dir.iterdir())
        if path.is_file()
    }

    parsed = parsers.parse_epw_imag_iso(file_contents, prefix="aiida")

    assert parsed, "No temperatures were parsed from imag_iso files."

    regression_data = {T: parsed[T].tolist()[:10] for T in sorted(parsed.keys())}
    data_regression.check(regression_data)


def test_parse_epw_imag_aniso_gap0(files_path: Path, data_regression):
    """Parse anisotropic ``imag_aniso_gap0`` files from a folder mapping."""
    aniso_dir = files_path / "tools" / "parsers" / "fsr_aniso_eliashberg"

    file_contents = {
        path.name: path.read_text()
        for path in sorted(aniso_dir.iterdir())
        if path.is_file()
    }

    parsed = parsers.parse_epw_imag_aniso_gap0(file_contents, prefix="aiida")

    assert parsed, "No temperatures were parsed from imag_aniso_gap0 files."

    regression_data = {T: parsed[T].tolist()[:10] for T in sorted(parsed.keys())}
    data_regression.check(regression_data)


def test_parse_aniso_FS(files_path: Path):
    """Parse a real anisotropic gap FS file."""
    gap_fs_path = (
        files_path
        / "tools"
        / "parsers"
        / "fbw_aniso_eliashberg"
        / "MgB2.imag_aniso_gap_FS_015.00"
    )
    content = gap_fs_path.read_text()

    parsed = parsers.parse_aniso_FS(content)

    assert "units" in parsed
    assert parsed["units"] == {
        "kpoints": "crystal",
        "energy": "eV",
        "delta": "meV",
    }

    # Verify band 1 and band 2 are present
    assert 1 in parsed
    assert 2 in parsed

    # Verify shapes
    assert parsed[1]["kpoints"].ndim == 2
    assert parsed[1]["kpoints"].shape[1] == 3
    assert parsed[1]["energy"].ndim == 1
    assert parsed[1]["delta"].ndim == 1

    assert parsed[2]["kpoints"].ndim == 2
    assert parsed[2]["kpoints"].shape[1] == 3
    assert parsed[2]["energy"].ndim == 1
    assert parsed[2]["delta"].ndim == 1

    # Verify first row values for band 1
    #     0.000000    0.000000    0.000000       1    0.347134      15.005356995134328
    assert parsed[1]["kpoints"][0].tolist() == [0.0, 0.0, 0.0]
    assert parsed[1]["energy"][0] == pytest.approx(0.347134)
    assert parsed[1]["delta"][0] == pytest.approx(15.005356995134328)

    # Verify first row values for band 2
    #     0.000000    0.000000    0.000000       2    0.347134      15.005356995134328
    assert parsed[2]["kpoints"][0].tolist() == [0.0, 0.0, 0.0]
    assert parsed[2]["energy"][0] == pytest.approx(0.347134)
    assert parsed[2]["delta"][0] == pytest.approx(15.005356995134328)


def test_parse_aniso(files_path: Path):
    """Parse a real anisotropic Eliashberg imag_aniso file."""
    gap_path = (
        files_path
        / "tools"
        / "parsers"
        / "fbw_aniso_eliashberg"
        / "MgB2.imag_aniso_015.00"
    )
    content = gap_path.read_text()

    parsed = parsers.parse_aniso(content)

    assert "units" in parsed
    assert parsed["units"] == {
        "frequency": "eV",
        "energy": "eV",
        "znorm": "",
        "delta": "eV",
        "shift": "eV",
    }

    # Verify a few frequency keys exist (e.g. the first one: 4.0608226305e-03)
    f1 = 4.0608226305e-03
    assert f1 in parsed

    snapshot = parsed[f1]
    assert snapshot["energy"].ndim == 1
    assert snapshot["znorm"].ndim == 1
    assert snapshot["delta"].ndim == 1
    assert snapshot["shift"].ndim == 1

    # Equal lengths
    n_kpts = snapshot["energy"].shape[0]
    assert snapshot["znorm"].shape[0] == n_kpts
    assert snapshot["delta"].shape[0] == n_kpts
    assert snapshot["shift"].shape[0] == n_kpts

    # Verify first row values of f1
    # 4.0608226305E-03    3.4713418718E-01    2.3199673459E+00    1.5005356995E-02    3.3472332952E-03
    assert snapshot["energy"][0] == pytest.approx(0.34713418718)
    assert snapshot["znorm"][0] == pytest.approx(2.3199673459)
    assert snapshot["delta"][0] == pytest.approx(15.005356995e-3)
    assert snapshot["shift"][0] == pytest.approx(3.3472332952e-3)


def test_parser_robust_exception_handling():
    """Test that robust parsing exception handling throws clear ValueError."""
    import pytest

    # 1. bands
    with pytest.raises(ValueError, match="Malformed bands file"):
        parsers.parse_epw_bands("invalid header content")

    # 2. a2f
    with pytest.raises(
        ValueError,
        match="Malformed .a2f file: Could not parse the number of smearing values",
    ):
        parsers.parse_epw_a2f("invalid content")

    with pytest.raises(
        ValueError, match="Malformed .a2f file: The a2F spectrum table is empty"
    ):
        parsers.parse_epw_a2f(
            " w[meV] a2f and integrated 2*a2f/w for   10 smearing values\n Integrated el-ph coupling"
        )

    # 3. max_eigenvalue
    with pytest.raises(
        ValueError, match="Finish: Solving \\(isotropic\\) linearized Eliashberg"
    ):
        parsers.parse_epw_max_eigenvalue("some content")

    # 4. eldos
    with pytest.raises(ValueError, match="Malformed electronic DOS file"):
        parsers.parse_epw_dos("not float content")

    # 5. phdos
    with pytest.raises(ValueError, match="Malformed phonon DOS file"):
        parsers.parse_epw_phdos("not float content")

    with pytest.raises(
        ValueError,
        match="Could not parse the number of smearing values from the header",
    ):
        parsers.parse_epw_phdos("w[meV] phdos[states/meV]\n0.1 1.0")

    # 6. imag_aniso_gap_FS
    with pytest.raises(ValueError, match="Malformed imag_aniso_gap_FS file"):
        parsers.parse_aniso_FS("not float content")

    # 7. imag_aniso
    with pytest.raises(ValueError, match="Malformed imag_aniso file"):
        parsers.parse_aniso("not float content")
