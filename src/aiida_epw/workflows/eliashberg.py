"""Adaptive work chain for Eliashberg temperature sampling."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, append_, while_

from aiida_epw.common.types import CalculationTypes
from aiida_epw.tools.eliashberg import (
    get_temperature_list,
    set_temperature_list,
    suggest_temperatures,
)
from aiida_epw.workflows.base import EpwBaseWorkChain


def extract_gap_series(outputs, is_anisotropic):
    """Return an iso/aniso representative gap series from EPW outputs."""
    if is_anisotropic:
        if "aniso_gap_functions" not in outputs:
            return None
        return outputs.aniso_gap_functions.get_averaged_gap()

    if "iso_gap_functions" not in outputs:
        return None
    return outputs.iso_gap_functions.get_gap_FS()


class EliashbergWorkChain(WorkChain):
    """Run EPW Eliashberg calculations and adapt the temperature sampling."""

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.expose_inputs(EpwBaseWorkChain, exclude=("calculation_type",))
        spec.input(
            "adaptive",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Whether to refine the temperature list from parsed gap data.",
        )
        spec.input(
            "max_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(4),
            help="Maximum number of EPW calculations for adaptive sampling.",
        )
        spec.input(
            "temps",
            valid_type=orm.List,
            required=False,
            help="Initial temperatures, in kelvin, for the Eliashberg sampling.",
        )
        spec.input(
            "sampling",
            valid_type=orm.Dict,
            required=False,
            help="Options controlling the gap-based temperature sampling heuristic.",
        )

        spec.outline(
            cls.setup,
            while_(cls.should_run_epw)(
                cls.run_epw,
                cls.inspect_epw,
            ),
            cls.results,
        )

        spec.expose_outputs(EpwBaseWorkChain)
        spec.output(
            "result",
            valid_type=orm.Dict,
            help="Summary of the adaptive temperature-sampling decisions.",
        )
        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_EPW",
            message="The `EpwBaseWorkChain` subprocess failed.",
        )
        spec.exit_code(
            402,
            "ERROR_GAP_OUTPUT_MISSING",
            message="The expected gap output was not produced.",
        )

    @classmethod
    def get_builder_from_protocol(
        cls,
        epw_code=None,
        parent_epw=None,
        protocol=None,
        overrides=None,
        options=None,
        parent_folder_epw=None,
        w90_chk_to_ukk_script=None,
        quadrupole_dir=None,
        protocol_filename="base.yaml",
        momentum_dependence=None,
        full_bandwidth=None,
        real_axis=None,
        analytical_continuation=None,
        adaptive=None,
        max_iterations=None,
        temps=None,
        sampling=None,
        code=None,
        **kwargs,
    ):
        """Return a builder prepopulated from a parent Wannier-representation EPW run."""
        if epw_code is None:
            epw_code = code
        if epw_code is None:
            raise ValueError("The `epw_code` input is required.")
        if parent_epw is None:
            raise ValueError("The `parent_epw` input is required.")

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

        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=epw_code,
            structure=structure,
            protocol=protocol,
            overrides=overrides,
            options=options,
            w90_chk_to_ukk_script=w90_chk_to_ukk_script,
            quadrupole_dir=quadrupole_dir,
            protocol_filename=protocol_filename,
            momentum_dependence=momentum_dependence,
            full_bandwidth=full_bandwidth,
            real_axis=real_axis,
            analytical_continuation=analytical_continuation,
            **kwargs,
        )
        epw_builder.kpoints = epw_source.inputs.kpoints
        epw_builder.qpoints = epw_source.inputs.qpoints
        epw_builder.parent_folder_epw = parent_folder_epw

        builder = cls.get_builder()
        for name in epw_builder:
            if name == "calculation_type":
                continue
            builder[name] = epw_builder[name]

        if adaptive is not None:
            builder.adaptive = orm.Bool(adaptive)
        if max_iterations is not None:
            builder.max_iterations = orm.Int(max_iterations)
        if temps is not None:
            builder.temps = orm.List(list=temps)
        if sampling is not None:
            builder.sampling = orm.Dict(dict=sampling)

        return builder

    def setup(self):
        """Initialize adaptive-sampling state."""
        self.ctx.iteration = 0
        self.ctx.should_run = True
        self.ctx.reports = []
        self.ctx.inputs = AttributeDict(self.exposed_inputs(EpwBaseWorkChain))
        self.ctx.inputs.calculation_type = orm.EnumData(CalculationTypes.ELIASHBERG)
        parameters = self.ctx.inputs.parameters.get_dict()
        temperatures = (
            [float(temperature) for temperature in self.inputs.temps.get_list()]
            if "temps" in self.inputs
            else get_temperature_list(parameters)
        )
        if temperatures:
            self.ctx.inputs.parameters = orm.Dict(
                set_temperature_list(parameters, temperatures)
            )

    def should_run_epw(self):
        """Return whether another EPW calculation should be submitted."""
        return self.ctx.should_run

    def run_epw(self):
        """Run one EPW calculation with the current explicit temperature list."""
        self.ctx.iteration += 1
        inputs = AttributeDict(self.ctx.inputs)
        inputs.setdefault("metadata", {})["call_link_label"] = (
            f"eliashberg_{self.ctx.iteration:02d}"
        )

        workchain_node = self.submit(EpwBaseWorkChain, **inputs)
        self.report(
            f"launching EpwBaseWorkChain<{workchain_node.pk}> for Eliashberg "
            f"sampling iteration {self.ctx.iteration}"
        )
        return {"epw": append_(workchain_node)}

    def inspect_epw(self):
        """Check the EPW subprocess status and update temperature sampling."""
        workchain = self.ctx.epw[-1]
        if not workchain.is_finished_ok:
            self.report(
                f"EpwBaseWorkChain<{workchain.pk}> failed with exit status "
                f"{workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_EPW

        gap_series = self._extract_gap_series(workchain)
        if gap_series is None:
            return self.exit_codes.ERROR_GAP_OUTPUT_MISSING

        parameters = self.ctx.inputs.parameters.get_dict()
        temperatures = get_temperature_list(parameters)
        sampling = self.inputs.sampling.get_dict() if "sampling" in self.inputs else {}
        report = suggest_temperatures(
            temperatures,
            gap_series,
            **sampling,
        )
        self.ctx.reports.append(report)

        should_stop = (
            not self.inputs.adaptive.value
            or report["status"] in ("GOOD", "BAD_DATA")
            or self.ctx.iteration >= self.inputs.max_iterations.value
            or report["temperatures"] == temperatures
        )
        if should_stop:
            self.ctx.should_run = False
            return

        self.report(
            f"temperature sampling status `{report['status']}`: {report['reason']} "
            f"Next temperatures: {report['temperatures']}"
        )
        self.ctx.inputs.parameters = orm.Dict(
            set_temperature_list(parameters, report["temperatures"])
        )

    def results(self):
        """Expose the outputs of the final EPW subprocess and the sampling report."""
        final_workchain = self.ctx.epw[-1]
        self.out_many(self.exposed_outputs(final_workchain, EpwBaseWorkChain))
        self.out(
            "result",
            orm.Dict(
                {
                    "iterations": self.ctx.iteration,
                    "reports": self.ctx.reports,
                    "final_temperatures": get_temperature_list(
                        self.ctx.inputs.parameters.get_dict()
                    ),
                }
            ),
        )

    def _extract_gap_series(self, workchain):
        """Return an iso/aniso representative gap series from a finished subprocess."""
        momentum_dependence = self.ctx.inputs.get("momentum_dependence")
        is_anisotropic = momentum_dependence is not None and momentum_dependence.value

        return extract_gap_series(workchain.outputs, is_anisotropic)
