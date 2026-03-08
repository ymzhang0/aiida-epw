"""Manual parsing functions for post-processing."""

import io
import re

import numpy

Ry2eV = 13.605662285137


def parse_epw_bands(file_content):
    """Parse the contents of a `band.eig`-style EPW bands file."""
    nbnd, _ = (
        int(value)
        for value in re.search(
            r"&plot nbnd=\s+(\d+), nks=\s+(\d+)", file_content
        ).groups()
    )
    kpt_pattern = re.compile(r"^\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s*$")
    band_pattern = re.compile(r"\s+([-\d\.]+)" * nbnd)

    parsed_data = {"kpoints": [], "bands": []}

    for line in file_content.splitlines():
        match_kpt = re.search(kpt_pattern, line)
        if match_kpt:
            parsed_data["kpoints"].append(match_kpt.groups())
            continue

        match_band = re.search(band_pattern, line)
        if match_band:
            parsed_data["bands"].append(match_band.groups())

    parsed_data["kpoints"] = numpy.array(parsed_data["kpoints"], dtype=float)
    parsed_data["bands"] = numpy.array(parsed_data["bands"], dtype=float)

    return parsed_data


def parse_epw_a2f(file_content):
    """Parse the contents of the `.a2f` file."""
    parsed_data = {}

    a2f, footer = file_content.split("\n Integrated el-ph coupling", maxsplit=1)

    a2f_array = numpy.array(
        [line.split() for line in a2f.splitlines()[1:] if line.strip()],
        dtype=float,
    )
    parsed_data["frequency"] = a2f_array[:, 0]
    parsed_data["a2f"] = a2f_array[:, 1:]

    footer = footer.split("\n")
    parsed_data["lambda"] = numpy.array(footer[1].strip("# ").split(), dtype=float)
    parsed_data["phonon_smearing"] = numpy.array(
        footer[3].strip("# ").split(), dtype=float
    )

    key_property_dict = {
        "Electron smearing (eV)": "electron_smearing",
        "Fermi window (eV)": "fermi_window",
        "Summed el-ph coupling": "summed_elph_coupling",
    }
    for line in footer:
        for key, property_name in key_property_dict.items():
            if key in line:
                parsed_data[property_name] = float(line.split()[-1])

    return parsed_data


def parse_epw_max_eigenvalue(file_content):
    """Parse the max_eigenvalue part of the `stdout` file when solving the linearized Eliashberg equation."""
    parsed_data = {}
    re_pattern = re.compile(r"\s+([\d\.]+)\s+([\d\.-]+)\s+\d+\s+[\d\.]+\s+\d+\n")
    parsing_block = file_content.split(
        "Finish: Solving (isotropic) linearized Eliashberg"
    )[0]

    parsed_data["max_eigenvalue"] = numpy.array(
        re_pattern.findall(parsing_block), dtype=float
    )
    return parsed_data


def parse_epw_eldos(file_content):
    """Parse the contents of the electronic DOS file produced by EPW."""
    dos = numpy.loadtxt(io.StringIO(file_content), dtype=float, comments="#")
    return {
        "energy": dos[:, 0],
        "edos": dos[:, 1],
        "integrated_dos": dos[:, 2],
    }


def parse_epw_phdos(file_content):
    """Parse the contents of the phonon DOS file produced by EPW."""
    phdos = numpy.loadtxt(io.StringIO(file_content), dtype=float, skiprows=1)
    return {
        "frequency": phdos[:, 0],
        "phdos": phdos[:, 1:],
    }


def parse_epw_imag_iso(file_contents, prefix="aiida"):
    """Parse the isotropic gap functions from EPW isotropic Eliashberg equation calculation.

    :param file_contents: mapping of file names to file contents.
    :param prefix: the prefix of the `imag_iso` files.
    :returns: dictionary containing the isotropic gap functions keyed by temperature.
    """
    parsed_data = {}
    pattern_iso = re.compile(rf"^{prefix}\.imag_iso_(\d{{3}}\.\d{{2}})$")

    for filename, file_content in file_contents.items():
        match = pattern_iso.match(filename)
        if match:
            temperature = float(match.group(1))
            gap_function = numpy.loadtxt(
                io.StringIO(file_content), dtype=float, comments="#", skiprows=1
            )
            parsed_data[temperature] = gap_function
    return parsed_data


def parse_epw_imag_aniso_gap0(file_contents, prefix="aiida"):
    """Parse the anisotropic gap functions from EPW anisotropic Eliashberg equation calculation.

    :param file_contents: mapping of file names to file contents.
    :param prefix: the prefix of the `imag_aniso_gap0` files.
    :returns: dictionary containing the anisotropic gap functions keyed by temperature.
    """
    parsed_data = {}
    pattern_aniso_gap0 = re.compile(rf"^{prefix}\.imag_aniso_gap0_(\d{{3}}\.\d{{2}})$")

    for filename, file_content in file_contents.items():
        match = pattern_aniso_gap0.match(filename)
        if match:
            temperature = float(match.group(1))
            gap_function = numpy.loadtxt(
                io.StringIO(file_content), dtype=float, comments="#", skiprows=1
            )
            parsed_data[temperature] = gap_function
    return parsed_data
