"""Manual parsing functions for post-processing."""

import numpy
import re
import io

Ry2eV = 13.605662285137


def parse_epw_a2f(file_content):
    """Parse the contents of the `.a2f` file."""
    parsed_data = {}

    a2f, footer = file_content.split("\n Integrated el-ph coupling")

    a2f_array = numpy.array([line.split() for line in a2f.split("\n")], dtype=float)
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
        for key, property in key_property_dict.items():
            if key in line:
                parsed_data[property] = float(line.split()[-1])

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


def parse_epw_imag_iso(file_contents, prefix="aiida"):
    """Parse the isotropic gap functions from EPW isotropic Eliashberg equation calculation.
    parameters:
        folder: the folder containing the `imag_iso` files. When serving as a helper function, it can take a `Retrieved` folder from aiida .
        When used independently, it can take a local folder.
        prefix: the prefix of the `imag_iso` files.
    returns:
        parsed_data: a dictionary containing the isotropic gap functions of numpy array type and the corresponding temperatures as keys.
    """
    parsed_data = {}
    pattern_iso = re.compile(rf"^{prefix}\.imag_iso_(\d{{3}}\.\d{{2}})$")

    for filename, file_content in file_contents.items():
        match = pattern_iso.match(filename)
        if match:
            T = float(match.group(1))
            gap_function = numpy.loadtxt(
                io.StringIO(file_content), dtype=float, comments="#", skiprows=1
            )
            parsed_data[T] = gap_function
    return parsed_data


def parse_epw_imag_aniso_gap0(file_contents, prefix="aiida"):
    """Parse the anisotropic gap functions from EPW anisotropic Eliashberg equation calculation.
    parameters:
        file_contents: a dictionary containing the file contents with filename as keys.
        prefix: the prefix of the `imag_aniso_gap0` files.
    returns:
        parsed_data: a sorted dictionary containing the anisotropic gap functions of numpy array type and the corresponding temperatures as keys.
    """
    parsed_data = {}
    pattern_aniso_gap0 = re.compile(rf"^{prefix}\.imag_aniso_gap0_(\d{{3}}\.\d{{2}})$")

    for filename, file_content in file_contents.items():
        match = pattern_aniso_gap0.match(filename)
        if match:
            T = float(match.group(1))
            gap_function = numpy.loadtxt(
                io.StringIO(file_content), dtype=float, comments="#", skiprows=1
            )
            parsed_data[T] = gap_function
    return parsed_data
