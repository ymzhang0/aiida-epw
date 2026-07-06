"""Work chain for converging the smearing parameter degaussw using parallel EpwBaseWorkChains."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_epw.workflows.base import EpwBaseWorkChain


class EpwDegausswConvWorkChain(ProtocolMixin, WorkChain):
    """This workchain runs a series of parallel `EpwBaseWorkChain`s in interpolation mode
    with different `degaussw` (smearing width) values on a fixed fine k/q-mesh,
    and converges the critical temperature (or other properties)."""

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="Structure used for the calculations.",
        )
        spec.input(
            "parent_folder_epw",
            valid_type=(orm.RemoteData, orm.RemoteStashFolderData),
            help="Remote folder containing the Wannier representation from a previous coarse grid run.",
        )
        spec.input(
            "degaussw_list",
            valid_type=orm.List,
            help="List of degaussw values (in eV) to run and check for convergence.",
        )
        spec.input(
            "convergence_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.05),
            help="Relative threshold for critical temperature convergence.",
        )

        spec.expose_inputs(
            EpwBaseWorkChain,
            namespace="epw",
            exclude=(
                "clean_workdir",
                "parent_folder_ph",
                "parent_folder_nscf",
                "parent_folder_chk",
            ),
            namespace_options={
                "help": "Inputs forwarded to `EpwBaseWorkChain` for the individual runs."
            },
        )

        spec.outline(
            cls.setup,
            cls.run_parallel_degaussw,
            cls.inspect_convergence,
            cls.results,
        )

        spec.output(
            "parameters",
            valid_type=orm.Dict,
            help="The output parameters of the converged calculation.",
        )
        spec.output(
            "converged_degaussw",
            valid_type=orm.Float,
            help="The converged degaussw value.",
        )

        spec.exit_code(
            401,
            "ERROR_ALL_SUB_PROCESSES_FAILED",
            message="All EpwBaseWorkChain sub-processes failed.",
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from . import protocols

        return files(protocols) / "degaussw.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        epw_code,
        parent_epw,
        protocol=None,
        overrides=None,
        parent_folder_epw=None,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()

        if parent_epw.process_label == "EpwPrepWorkChain":
            epw_source = (
                parent_epw.base.links.get_outgoing(link_label_filter="epw_base")
                .first()
                .node
            )
            structure = parent_epw.inputs.structure
        elif parent_epw.process_label == "EpwBaseWorkChain":
            epw_source = parent_epw
            try:
                structure = parent_epw.inputs.structure
            except AttributeError as exc:
                raise ValueError(
                    "The `parent_epw` (EpwBaseWorkChain) does not contain `structure` in its inputs."
                ) from exc
        else:
            raise ValueError(f"Invalid parent_epw process: {parent_epw.process_label}")

        if parent_folder_epw is None:
            if epw_source.inputs.code.computer.hostname != epw_code.computer.hostname:
                raise ValueError(
                    "The `epw_code` must be configured on the same computer as that where the `parent_epw` was run."
                )
            parent_folder_epw = epw_source.outputs.remote_stash
        else:
            # TODO: Add check to make sure parent_folder_epw is on same computer as epw_code
            pass

        epw_inputs = inputs.get("epw", None)

        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=epw_code,
            structure=structure,
            protocol=protocol,
            overrides=epw_inputs,
        )

        epw_builder.kpoints = epw_source.inputs.kpoints
        epw_builder.qpoints = epw_source.inputs.qpoints

        if "settings" in epw_inputs:
            epw_builder.settings = orm.Dict(epw_inputs["settings"])

        builder.epw = epw_builder

        if isinstance(inputs["degaussw_list"], list):
            builder.degaussw_list = orm.List(inputs["degaussw_list"])

        builder.convergence_threshold = orm.Float(inputs["convergence_threshold"])
        builder.structure = structure
        builder.parent_folder_epw = parent_folder_epw

        return builder

    def setup(self):
        """Setup context variables."""
        degaussw_list = self.inputs.degaussw_list.get_list()
        # Sort in descending order (largest smearing to smallest)
        degaussw_list.sort(reverse=True)
        self.ctx.degaussw_values = degaussw_list

    def run_parallel_degaussw(self):
        """Submit calculations for all degaussw values in parallel."""
        base_inputs = AttributeDict(
            self.exposed_inputs(EpwBaseWorkChain, namespace="epw")
        )
        base_inputs.structure = self.inputs.structure
        base_inputs.parent_folder_epw = self.inputs.parent_folder_epw

        running_workchains = {}
        for idx, degaussw in enumerate(self.ctx.degaussw_values):
            inputs = AttributeDict(base_inputs)
            # Update the degaussw value in inputs.parameters["INPUTEPW"]
            parameters = base_inputs.parameters.get_dict()
            parameters.setdefault("INPUTEPW", {})["degaussw"] = degaussw
            inputs.parameters = orm.Dict(parameters)

            inputs.setdefault("metadata", {})["call_link_label"] = (
                f"degaussw_{idx:02d}_{str(degaussw).replace('.', '_')}"
            )

            workchain_node = self.submit(EpwBaseWorkChain, **inputs)
            label = f"wc_{idx}"
            running_workchains[label] = workchain_node

            self.report(
                f"launching EpwBaseWorkChain<{workchain_node.pk}> with degaussw = {degaussw} eV"
            )

        return ToContext(**running_workchains)

    def inspect_convergence(self):
        """Analyze results from the parallel runs and find the converged one."""
        results = []
        for idx, degaussw in enumerate(self.ctx.degaussw_values):
            label = f"wc_{idx}"
            workchain = getattr(self.ctx, label)
            if not workchain.is_finished_ok:
                self.report(
                    f"EpwBaseWorkChain<{workchain.pk}> with degaussw = {degaussw} failed "
                    f"with exit status {workchain.exit_status}"
                )
                continue

            try:
                output_params = workchain.outputs.output_parameters.get_dict()
                tc = output_params.get("Allen_Dynes_Tc", None)
            except AttributeError:
                tc = None

            results.append(
                {
                    "degaussw": degaussw,
                    "tc": tc,
                    "workchain": workchain,
                    "pk": workchain.pk,
                }
            )

        if not results:
            self.report("All sub-workchains failed.")
            return self.exit_codes.ERROR_ALL_SUB_PROCESSES_FAILED

        # Sort results by degaussw descending
        results.sort(key=lambda x: x["degaussw"], reverse=True)

        self.report("Degaussw convergence scan results:")
        for res in results:
            self.report(
                f"  degaussw = {res['degaussw']} eV -> Allen-Dynes Tc = {res['tc']} K (PK: {res['pk']})"
            )

        converged_index = None
        threshold = self.inputs.convergence_threshold.value

        # Compare consecutive entries (from largest degaussw to smallest)
        for i in range(len(results) - 1):
            tc_prev = results[i]["tc"]
            tc_curr = results[i + 1]["tc"]

            if tc_prev is not None and tc_curr is not None and tc_curr != 0:
                rel_diff = abs(tc_prev - tc_curr) / tc_curr
                if rel_diff <= threshold:
                    converged_index = i + 1
                    self.report(
                        f"Convergence achieved at degaussw = {results[converged_index]['degaussw']} eV "
                        f"(relative difference = {rel_diff:.4f} <= threshold {threshold})"
                    )
                    break

        if converged_index is None:
            self.report(
                "Convergence threshold was not met. Selecting the smallest degaussw calculation."
            )
            converged_index = len(results) - 1

        self.ctx.converged_res = results[converged_index]

    def results(self):
        """Expose outputs."""
        self.out(
            "parameters", self.ctx.converged_res["workchain"].outputs.output_parameters
        )
        self.out("converged_degaussw", orm.Float(self.ctx.converged_res["degaussw"]))
