import re
from pathlib import Path

import numpy
from aiida import orm
from aiida_quantumespresso.parsers.base import BaseParser
from aiida_quantumespresso.utils.mapping import get_logging_container
from packaging.version import Version

from aiida_epw.calculations.epw import EpwCalculation
from aiida_epw.data import (
    A2fData,
    PA2fData,
    DosData,
    PDosData,
    PhDosData,
    GapFunctionData,
    LambdaFSData,
)
from aiida_epw.tools.parsers import (
    parse_epw_a2f,
    parse_epw_a2f_proj,
    parse_epw_bands,
    parse_epw_dos,
    parse_epw_imag_aniso_gap0,
    parse_epw_imag_iso,
    parse_epw_lambda_fs,
    parse_epw_lambda_k_pairs,
    parse_epw_max_eigenvalue,
    parse_epw_phdos,
    parse_epw_phdos_proj,
    parse_aniso_gap_FS,
    parse_aniso,
)
from aiida_epw.parsers.schemas import (
    REGEX_PATTERNS_LEGACY,
    REGEX_PATTERNS_MODERN,
    parse_fortran_float,
)


class EpwParser(BaseParser):
    """``Parser`` implementation for the ``EpwCalculation`` calculation job."""

    success_string = "EPW.bib"

    class_error_map = {
        "Size of required memory exceeds max_memlt": "ERROR_MEMORY_EXCEEDS_MAX_MEMLT",
        "internal error, cannot bracket Ef": "ERROR_CANNOT_BRACKET_EF",
        r"Error in routine mix_broyden \(\d+\):\s*factorization": "ERROR_FACTORIZATION",
    }

    @staticmethod
    def is_scheduler_out_of_memory(stderr):
        """Return whether scheduler stderr indicates an out-of-memory kill."""
        if not stderr:
            return False

        stderr_lower = stderr.lower()
        return any(
            marker in stderr_lower
            for marker in (
                "oom_kill",
                "out of memory",
                "oom killed",
                "exceeded memory limit",
            )
        )

    @staticmethod
    def get_parser_settings_key():
        """Return the settings key reserved for parser-specific options."""
        return "parser_options"

    def get_retrieved_content(self, *filenames):
        """Return the content of the first retrieved file that exists."""
        for filename in filenames:
            try:
                return self.retrieved.base.repository.get_object_content(filename)
            except FileNotFoundError:
                continue

        return None

    def get_retrieved_contents_matching(self, pattern):
        """Return retrieved file contents whose names match a compiled regex pattern."""
        return {
            filename: self.retrieved.base.repository.get_object_content(filename)
            for filename in self.retrieved.base.repository.list_object_names()
            if pattern.match(filename)
        }

    def parse(self, **kwargs):
        """Parse the retrieved files of a completed ``EpwCalculation`` into output nodes."""
        logs = get_logging_container()

        stdout, parsed_data, logs = self.parse_stdout_from_retrieved(logs)
        scheduler_stderr = self.node.get_scheduler_stderr()

        # Preserve scheduler walltime failures instead of overriding them with parser-side stdout errors.
        if (
            self.node.exit_status
            == self.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME.status
        ):
            return self.exit(logs=logs)

        if (
            self.node.exit_status
            == self.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY.status
        ):
            return self.exit(logs=logs)

        if self.is_scheduler_out_of_memory(scheduler_stderr):
            return self.exit(self.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY, logs)

        base_exit_code = self.check_base_errors(logs)
        if base_exit_code:
            return self.exit(base_exit_code, logs)

        parsed_epw, logs = self.parse_stdout(
            stdout, logs, code_version=Version(parsed_data["code_version"])
        )
        parsed_data.update(parsed_epw)

        elbands_contents = self.get_retrieved_content(
            EpwCalculation._output_elbands_file
        )
        if elbands_contents is not None:
            self.out(
                "el_band_structure",
                self.parse_bands(
                    elbands_contents, getattr(self.node.inputs, "kfpoints", None), "eV"
                ),
            )

        phbands_contents = self.get_retrieved_content(
            EpwCalculation._output_phbands_file
        )
        if phbands_contents is not None:
            self.out(
                "ph_band_structure",
                self.parse_bands(
                    phbands_contents, getattr(self.node.inputs, "qfpoints", None), "meV"
                ),
            )

        a2f_contents = self.get_retrieved_content(EpwCalculation._OUTPUT_A2F_FILE)
        if a2f_contents is not None:
            a2f_data, parsed_a2f = self.parse_a2f(a2f_contents)
            self.out("a2f", a2f_data)
            parsed_data.update(parsed_a2f)

        dos_contents = self.get_retrieved_content(
            EpwCalculation._OUTPUT_DOS_FILE,
            Path(
                EpwCalculation._OUTPUT_SUBFOLDER, EpwCalculation._OUTPUT_DOS_FILE
            ).as_posix(),
        )
        if dos_contents is not None:
            self.out("dos", self.parse_dos(dos_contents))

        phdos_contents = self.get_retrieved_content(EpwCalculation._OUTPUT_PHDOS_FILE)
        if phdos_contents is not None:
            self.out("phdos", self.parse_phdos(phdos_contents))

        phdos_proj_contents = self.get_retrieved_content(
            EpwCalculation._OUTPUT_PHDOS_PROJ_FILE
        )
        if phdos_proj_contents is not None:
            self.out("phdos_proj", self.parse_phdos_proj(phdos_proj_contents))

        a2f_proj_contents = self.get_retrieved_content(
            EpwCalculation._OUTPUT_A2F_PROJ_FILE
        )
        if a2f_proj_contents is not None:
            self.out("a2f_proj", self.parse_a2f_proj(a2f_proj_contents))

        lambda_FS_contents = self.get_retrieved_content(
            EpwCalculation._OUTPUT_LAMBDA_FS_FILE
        )
        if lambda_FS_contents is not None:
            self.out("lambda_FS", self.parse_lambda_FS(lambda_FS_contents))

        lambda_k_pairs_contents = self.get_retrieved_content(
            EpwCalculation._OUTPUT_LAMBDA_K_PAIRS_FILE
        )
        if lambda_k_pairs_contents is not None:
            self.out(
                "lambda_k_pairs",
                self.parse_lambda_k_pairs(lambda_k_pairs_contents),
            )

        iso_gap_pattern = re.compile(rf"^{EpwCalculation._PREFIX}\.imag_iso_\d+\.\d+$")
        if self.retrieved and any(
            iso_gap_pattern.match(name) for name in self.retrieved.list_object_names()
        ):
            self.out(
                "iso_gap_functions",
                self.parse_iso_gap_functions(self.retrieved),
            )

        aniso_gap_pattern = re.compile(
            rf"^{EpwCalculation._PREFIX}\.imag_aniso_gap0_\d+\.\d+$"
        )
        if self.retrieved and any(
            aniso_gap_pattern.match(name) for name in self.retrieved.list_object_names()
        ):
            self.out(
                "aniso_gap_functions",
                self.parse_aniso_gap_functions(self.retrieved),
            )

        aniso_gap_fs_pattern = re.compile(
            rf"^{EpwCalculation._PREFIX}\.imag_aniso_gap_FS_\d+\.\d+$"
        )
        if self.retrieved and any(
            aniso_gap_fs_pattern.match(name)
            for name in self.retrieved.list_object_names()
        ):
            self.out("aniso_gap_FS", self.parse_aniso_gap_fs(self.retrieved))

        aniso_imag_pattern = re.compile(
            rf"^{EpwCalculation._PREFIX}\.imag_aniso_\d+\.\d+$"
        )
        if self.retrieved and any(
            aniso_imag_pattern.match(name)
            for name in self.retrieved.list_object_names()
        ):
            fbw_enabled = False
            if "parameters" in self.node.inputs:
                params_dict = self.node.inputs.parameters.get_dict()
                fbw_enabled = params_dict.get("INPUTEPW", {}).get("fbw", False)
            restriction = "fbw" if fbw_enabled else "fsr"
            self.out(
                "aniso_gap_imag",
                self.parse_aniso_imag(self.retrieved, restriction=restriction),
            )

        if "max_eigenvalue" in parsed_data:
            self.out("max_eigenvalue", parsed_data.pop("max_eigenvalue"))

        if "Allen_Dynes_Tc" in parsed_data:
            parsed_data.setdefault("allen_dynes", parsed_data["Allen_Dynes_Tc"])

        self.out("output_parameters", orm.Dict(parsed_data))

        for exit_code in list(self.get_error_map().values()):
            if exit_code in logs.error:
                return self.exit(self.exit_codes.get(exit_code), logs)

        if "ERROR_OUTPUT_STDOUT_INCOMPLETE" in logs.error:
            return self.exit(
                self.exit_codes.get("ERROR_OUTPUT_STDOUT_INCOMPLETE"), logs
            )

        return self.exit(logs=logs)

    @staticmethod
    def parse_stdout(stdout, logs, code_version):
        """Parse the ``stdout``."""

        def parse_max_eigenvalue(stdout_block):
            parsed_max_ev = parse_epw_max_eigenvalue(stdout_block)
            max_eigenvalue_array = orm.XyData()
            max_eigenvalue_array.set_array(
                "max_eigenvalue",
                parsed_max_ev["max_eigenvalue"],
            )
            return max_eigenvalue_array

        patterns = (
            REGEX_PATTERNS_LEGACY
            if code_version < Version("5.9")
            else REGEX_PATTERNS_MODERN
        )

        data_block_marker_parser = (
            (
                "max_eigenvalue",
                "Superconducting transition temp. Tc",
                parse_max_eigenvalue,
            ),
        )
        parsed_data = {}
        stdout_lines = stdout.split("\n")

        for line_number, line in enumerate(stdout_lines):
            for key, type_func, pattern in patterns:
                match = pattern.search(line)
                if match:
                    parsed_data[key] = type_func(match.group(1))

            for (
                data_key,
                data_marker,
                block_parser,
            ) in data_block_marker_parser:
                if data_marker in line:
                    parsed_data[data_key] = block_parser(
                        "\n".join(stdout_lines[line_number:])
                    )

        # Parse carrier mobility matrices (SERTA and iBTE)
        from aiida_epw.tools.parsers import parse_transport_matrices
        import numpy

        # Identify SERTA block
        serta_match = re.search(
            r"BTE in the self-energy relaxation time approximation \(SERTA\)", stdout
        )
        # Identify BTE block (looking for standalone BTE header)
        bte_match = re.search(r"\n\s+BTE\s*\n", stdout)

        serta_idx = serta_match.start() if serta_match else -1
        bte_idx = bte_match.start() if bte_match else -1

        if serta_idx != -1:
            end_serta = bte_idx if bte_idx > serta_idx else len(stdout)
            serta_block = stdout[serta_idx:end_serta]
            serta_data = parse_transport_matrices(serta_block)

            for k, v in serta_data.items():
                parsed_data[f"serta_{k}"] = v

            # Maintain backward compatibility for mobility scalar
            if "mobility" in serta_data:
                parsed_data["mobility_SERTA"] = (
                    numpy.trace(numpy.array(serta_data["mobility"])) / 3.0
                )

        if bte_idx != -1:
            ibte_block = stdout[bte_idx:]
            ibte_data = parse_transport_matrices(ibte_block)

            for k, v in ibte_data.items():
                parsed_data[f"ibte_{k}"] = v

            # Maintain backward compatibility for mobility scalar
            if "mobility" in ibte_data:
                parsed_data["mobility_iBTE"] = (
                    numpy.trace(numpy.array(ibte_data["mobility"])) / 3.0
                )

        # Parse isotropic Eliashberg temperature blocks
        eliashberg_marker = "Solve isotropic Eliashberg equations"
        eliashberg_idx = stdout.find(eliashberg_marker)
        if eliashberg_idx != -1:
            eliashberg_content = stdout[eliashberg_idx:]
            temp_pattern = re.compile(r"temp\(\s*\d+\s*\)\s*=\s*([\d\.]+)\s*K")
            matches = list(temp_pattern.finditer(eliashberg_content))

            blocks = []
            for i, match in enumerate(matches):
                start = match.start()
                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(eliashberg_content)
                )

                unfolding_idx = eliashberg_content.find(
                    "Unfolding on the coarse grid", start, end
                )
                if unfolding_idx != -1:
                    end = unfolding_idx

                block_text = eliashberg_content[start:end]
                temp = float(match.group(1))

                nsiw_match = re.search(
                    r"Total number of frequency points nsiw\(\s*\d+\s*\)\s*=\s*(\d+)",
                    block_text,
                )
                wscut_match = re.search(
                    r"Cutoff frequency wscut\s*=\s*([\d\.]+)\s*eV", block_text
                )
                broyden_match = re.search(
                    r"broyden mixing factor\s*=\s*([\d\.]+)", block_text
                )
                nsiter_match = re.search(
                    r"Convergence was reached in nsiter\s*=\s*(\d+)", block_text
                )
                free_energy_match = re.search(
                    r"Free energy\s*=\s*([\d\.-]+)\s*meV", block_text
                )

                block_data = {"temp": temp}
                if nsiw_match:
                    block_data["nsiw"] = int(nsiw_match.group(1))
                if wscut_match:
                    block_data["wscut"] = float(wscut_match.group(1))
                if broyden_match:
                    block_data["broyden_mixing_factor"] = float(broyden_match.group(1))
                if nsiter_match:
                    block_data["nsiter"] = int(nsiter_match.group(1))
                if free_energy_match:
                    block_data["free_energy"] = float(free_energy_match.group(1))

                iw_match = re.search(
                    r"startiw\s*=\s*(\d+),\s*lastiw\s*=\s*(\d+),\s*nsiw\(itemp\)\s*=\s*(\d+)",
                    block_text,
                )
                if iw_match:
                    block_data["startiw"] = int(iw_match.group(1))
                    block_data["lastiw"] = int(iw_match.group(2))
                    block_data["nsiw_itemp"] = int(iw_match.group(3))

                iter_header = re.search(
                    r"iter\s+ethr\s+znormi\s+deltai\s+\[meV\]", block_text
                )
                if iter_header:
                    header_end = iter_header.end()
                    remaining_text = block_text[header_end:]
                    ethr_list = []
                    znormi_list = []
                    deltai_list = []
                    row_pattern = re.compile(
                        r"^\s*(\d+)\s+([\d\.\+\-EeDd]+)\s+([\d\.\+\-EeDd]+)\s+([\d\.\+\-EeDd]+)\s*$"
                    )
                    for line in remaining_text.split("\n"):
                        row_match = row_pattern.match(line)
                        if row_match:
                            ethr_list.append(parse_fortran_float(row_match.group(2)))
                            znormi_list.append(parse_fortran_float(row_match.group(3)))
                            deltai_list.append(parse_fortran_float(row_match.group(4)))
                        elif ethr_list:
                            break
                    if ethr_list:
                        block_data["iterations"] = {
                            "ethr": ethr_list,
                            "znormi": znormi_list,
                            "deltai": deltai_list,
                        }

                blocks.append(block_data)

            if blocks:
                parsed_data["isotropic_eliashberg"] = blocks

        # Parse anisotropic Eliashberg temperature blocks
        anisotropic_marker = "anisotropic Eliashberg equations"
        anisotropic_idx = stdout.find(anisotropic_marker)
        if anisotropic_idx != -1:
            anisotropic_content = stdout[anisotropic_idx:]
            temp_pattern = re.compile(r"temp\(\s*\d+\s*\)\s*=\s*([\d\.]+)\s*K")
            matches = list(temp_pattern.finditer(anisotropic_content))

            blocks = []
            for i, match in enumerate(matches):
                start = match.start()
                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(anisotropic_content)
                )

                unfolding_idx = anisotropic_content.find(
                    "Unfolding on the coarse grid", start, end
                )
                if unfolding_idx != -1:
                    end = unfolding_idx

                block_text = anisotropic_content[start:end]
                temp = float(match.group(1))

                nsiw_match = re.search(
                    r"Total number of frequency points nsiw\(\s*\d+\s*\)\s*=\s*(\d+)",
                    block_text,
                )
                wscut_match = re.search(
                    r"Cutoff frequency wscut\s*=\s*([\d\.]+)\s*eV", block_text
                )
                broyden_match = re.search(
                    r"broyden mixing factor\s*=\s*([\d\.]+)", block_text
                )
                nsiter_match = re.search(
                    r"Convergence was reached in nsiter\s*=\s*(\d+)", block_text
                )
                free_energy_match = re.search(
                    r"Free energy\s*=\s*([\d\.-]+)\s*meV", block_text
                )

                block_data = {"temp": temp}
                if nsiw_match:
                    block_data["nsiw"] = int(nsiw_match.group(1))
                if wscut_match:
                    block_data["wscut"] = float(wscut_match.group(1))
                if broyden_match:
                    block_data["broyden_mixing_factor"] = float(broyden_match.group(1))
                if nsiter_match:
                    block_data["nsiter"] = int(nsiter_match.group(1))
                if free_energy_match:
                    block_data["free_energy"] = float(free_energy_match.group(1))

                iw_match = re.search(
                    r"startiw\s*=\s*(\d+),\s*lastiw\s*=\s*(\d+),\s*nsiw\(itemp\)\s*=\s*(\d+)",
                    block_text,
                )
                if iw_match:
                    block_data["startiw"] = int(iw_match.group(1))
                    block_data["lastiw"] = int(iw_match.group(2))
                    block_data["nsiw_itemp"] = int(iw_match.group(3))

                gap_match = re.search(
                    r"Min\.\s*/\s*Max\.\s*values\s*of\s*superconducting\s*gap\s*=\s*([\d\.-]+)\s+([\d\.-]+)\s*meV",
                    block_text,
                )
                if gap_match:
                    block_data["gap_min"] = float(gap_match.group(1))
                    block_data["gap_max"] = float(gap_match.group(2))

                iter_header = re.search(
                    r"iter\s+ethr\s+znormi\s+deltai\s+\[meV\]", block_text
                )
                if iter_header:
                    header_end = iter_header.end()
                    remaining_text = block_text[header_end:]
                    ethr_list = []
                    znormi_list = []
                    deltai_list = []
                    shifti_list = []
                    mu_list = []
                    row_pattern = re.compile(
                        r"^\s*(\d+)\s+([\d\.\+\-EeDd]+)\s+([\d\.\+\-EeDd]+)\s+([\d\.\+\-EeDd]+)(?:\s+([\d\.\+\-EeDd]+)\s+([\d\.\+\-EeDd]+))?\s*$"
                    )
                    for line in remaining_text.split("\n"):
                        row_match = row_pattern.match(line)
                        if row_match:
                            ethr_list.append(parse_fortran_float(row_match.group(2)))
                            znormi_list.append(parse_fortran_float(row_match.group(3)))
                            deltai_list.append(parse_fortran_float(row_match.group(4)))
                            if row_match.group(5) is not None:
                                shifti_list.append(
                                    parse_fortran_float(row_match.group(5))
                                )
                            if row_match.group(6) is not None:
                                mu_list.append(parse_fortran_float(row_match.group(6)))
                        elif ethr_list:
                            break
                    if ethr_list:
                        iterations = {
                            "ethr": ethr_list,
                            "znormi": znormi_list,
                            "deltai": deltai_list,
                        }
                        if shifti_list:
                            iterations["shifti"] = shifti_list
                        if mu_list:
                            iterations["mu"] = mu_list
                        block_data["iterations"] = iterations

                blocks.append(block_data)

            if blocks:
                parsed_data["anisotropic_eliashberg"] = blocks

        return parsed_data, logs

    @staticmethod
    def parse_a2f(content):
        """Parse the contents of the `.a2f` file."""
        parsed_a2f = parse_epw_a2f(content)

        a2f_data = A2fData()
        a2f_data.set_a2f_data(
            frequency=parsed_a2f["frequency"],
            spectrum=parsed_a2f["a2f"],
            lambda_values=parsed_a2f["lambda"],
            phonon_smearing=parsed_a2f["phonon_smearing"],
            cumulative_lambda=parsed_a2f.get("cumulative_lambda"),
            electron_smearing=parsed_a2f.get("electron_smearing"),
            fermi_window=parsed_a2f.get("fermi_window"),
            summed_elph_coupling=parsed_a2f.get("summed_elph_coupling"),
        )

        parsed_data = {
            "degaussw": parsed_a2f["electron_smearing"],
            "fsthick": parsed_a2f["fermi_window"],
        }
        return a2f_data, parsed_data

    @staticmethod
    def parse_iso_gap_functions(file_contents):
        """Parse isotropic gap-function files into a typed datatype."""
        gap_functions = parse_epw_imag_iso(file_contents, prefix=EpwCalculation._PREFIX)
        gap_function_data = GapFunctionData()
        gap_function_data.set_gap_functions(gap_functions, kind="iso")
        return gap_function_data

    @staticmethod
    def parse_aniso_gap_functions(file_contents):
        """Parse anisotropic gap-function files into a typed datatype."""
        gap_functions = parse_epw_imag_aniso_gap0(
            file_contents, prefix=EpwCalculation._PREFIX
        )
        gap_function_data = GapFunctionData()
        gap_function_data.set_gap_functions(gap_functions, kind="aniso")
        return gap_function_data

    @staticmethod
    def parse_aniso_gap_fs(folder):
        """Parse the imag_aniso_gap_FS files into an ArrayData node."""
        parsed_by_temp = parse_aniso_gap_FS(folder, prefix=EpwCalculation._PREFIX)
        array_data = orm.ArrayData()
        temperatures = sorted(parsed_by_temp.keys())
        array_data.base.attributes.set("temperatures", temperatures)

        for temp in temperatures:
            parsed = parsed_by_temp[temp]
            temp_str = f"{temp:.2f}".replace(".", "_")
            bands = [int(k) for k in parsed.keys() if isinstance(k, int)]
            for band in sorted(bands):
                band_data = parsed[band]
                array_data.set_array(
                    f"temp_{temp_str}_band_{band}_kpoints", band_data["kpoints"]
                )
                array_data.set_array(
                    f"temp_{temp_str}_band_{band}_energy", band_data["energy"]
                )
                array_data.set_array(
                    f"temp_{temp_str}_band_{band}_delta", band_data["delta"]
                )
            array_data.base.attributes.set(f"temp_{temp_str}_bands", sorted(bands))
            if "units" in parsed:
                array_data.base.attributes.set(
                    f"temp_{temp_str}_units", parsed["units"]
                )
        return array_data

    @staticmethod
    def parse_aniso_imag(folder, restriction="fsr"):
        """Parse the imag_aniso files into an ArrayData node."""
        parsed_by_temp = parse_aniso(
            folder, prefix=EpwCalculation._PREFIX, restriction=restriction
        )
        array_data = orm.ArrayData()
        temperatures = sorted(parsed_by_temp.keys())
        array_data.base.attributes.set("temperatures", temperatures)

        for temp in temperatures:
            parsed = parsed_by_temp[temp]
            temp_str = f"{temp:.2f}".replace(".", "_")
            frequencies = sorted(
                [float(k) for k in parsed.keys() if isinstance(k, float)]
            )
            for index, w in enumerate(frequencies):
                w_data = parsed[w]
                array_data.set_array(
                    f"temp_{temp_str}_freq_{index}_energy", w_data["energy"]
                )
                array_data.set_array(
                    f"temp_{temp_str}_freq_{index}_znorm", w_data["znorm"]
                )
                array_data.set_array(
                    f"temp_{temp_str}_freq_{index}_delta", w_data["delta"]
                )
                if restriction == "fbw":
                    array_data.set_array(
                        f"temp_{temp_str}_freq_{index}_shift", w_data["shift"]
                    )
            array_data.base.attributes.set(f"temp_{temp_str}_frequencies", frequencies)
            if "units" in parsed:
                array_data.base.attributes.set(
                    f"temp_{temp_str}_units", parsed["units"]
                )
        return array_data

    @staticmethod
    def parse_a2f_proj(content):
        """Parse the contents of the `.a2f_proj` file."""
        parsed = parse_epw_a2f_proj(content)
        pa2f_data = PA2fData()
        pa2f_data.set_pa2f_data(
            frequency=parsed["frequency"],
            a2f=parsed["a2f"],
            projected_a2f=parsed["projected_a2f"],
            lambda_int=parsed.get("lambda_int"),
            lambda_sum=parsed.get("lambda_sum"),
        )
        return pa2f_data

    @staticmethod
    def parse_bands(content, kpoints_data, units):
        """Parse the contents of a band structure file."""
        parsed_bands = parse_epw_bands(content)
        kpts = parsed_bands["kpoints"]
        bands = parsed_bands["bands"]

        if kpoints_data is None:
            nbnd, nks = (
                int(v)
                for v in re.search(
                    r"&plot nbnd=\s+(\d+), nks=\s+(\d+)", content
                ).groups()
            )
            if len(kpts) != nks:
                raise ValueError(
                    "Could not reconstruct the band k-points from the retrieved EPW file."
                )

            kpoints_data = orm.KpointsData()
            kpoints_data.set_kpoints(kpts)

        bands_data = orm.BandsData()
        # We should use the KpointsData from the inputs.
        bands_data.set_kpointsdata(kpoints_data)
        bands_data.set_bands(bands, units=units)

        return bands_data

    @staticmethod
    def parse_dos(content):
        """Parse the contents of the `.dos` file."""
        parsed_dos = parse_epw_dos(content)
        dos_data = DosData()
        dos_data.set_dos_data(
            energy=parsed_dos["energy"],
            dos=parsed_dos["dos"],
            integrated_dos=parsed_dos.get("integrated_dos"),
        )
        return dos_data

    @staticmethod
    def parse_phdos(content):
        """Parse the contents of the `.phdos` file."""
        parsed_phdos = parse_epw_phdos(content)
        phdos_data = PhDosData()
        phdos_data.set_phdos_data(
            frequency=parsed_phdos["frequency"],
            phdos=parsed_phdos["phdos"],
        )
        return phdos_data

    @staticmethod
    def parse_phdos_proj(content):
        """Parse the contents of the `.phdos_proj` file."""
        parsed = parse_epw_phdos_proj(content)
        pdos_data = PDosData()
        pdos_data.set_pdos_data(
            frequency=parsed["frequency"],
            phdos=parsed["phdos"],
            projected_phdos=parsed["projected_phdos"],
        )
        return pdos_data

    @staticmethod
    def parse_lambda_FS(content):
        """Parse the contents of the `.lambda_FS` file."""
        parsed_lambda_fs = parse_epw_lambda_fs(content)
        lambda_fs_data = LambdaFSData()
        lambda_fs_data.set_lambda_fs(
            kpoints=parsed_lambda_fs["kpoints"],
            bands=parsed_lambda_fs["band"],
            energies=parsed_lambda_fs["energy"],
            couplings=parsed_lambda_fs["lambda"],
            energy_units=parsed_lambda_fs["energy_units"],
        )
        return lambda_fs_data

    @staticmethod
    def parse_lambda_k_pairs(content):
        """Parse the contents of the `.lambda_k_pairs` file."""
        parsed = parse_epw_lambda_k_pairs(content)
        dos_data = DosData()
        dos_data.set_dos_data(
            energy=parsed["energy"],
            dos=parsed["dos"],
            integrated_dos=parsed.get("integrated_dos"),
        )
        return dos_data

    @staticmethod
    def parse_gap_function(content, skiprows=0):
        """Parse the contents of the `gap_function.dat` file."""
        import io

        gap_function = numpy.loadtxt(
            io.StringIO(content), dtype=float, comments="#", skiprows=skiprows
        )

        return gap_function
