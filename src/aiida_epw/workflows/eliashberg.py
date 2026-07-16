"""Adaptive work chain for Eliashberg temperature sampling."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, append_, while_

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

        spec.expose_inputs(EpwBaseWorkChain)
        spec.input(
            "adaptive",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Whether to refine the temperature list from parsed gap data.",
        )
        spec.input(
            "max_sampling_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(4),
            help="Maximum number of EPW calculations for adaptive sampling.",
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

    def setup(self):
        """Initialize adaptive-sampling state."""
        self.ctx.iteration = 0
        self.ctx.should_run = True
        self.ctx.reports = []
        self.ctx.inputs = AttributeDict(self.exposed_inputs(EpwBaseWorkChain))
        parameters = self.ctx.inputs.parameters.get_dict()
        temperatures = get_temperature_list(parameters)
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
            or self.ctx.iteration >= self.inputs.max_sampling_iterations.value
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
