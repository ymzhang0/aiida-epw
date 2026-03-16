"""WorkGraph implementation of the superconductivity workflow."""

from __future__ import annotations

from typing import Any, Union

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida.engine.processes.builder import ProcessBuilderNamespace
from aiida_workgraph import If, WorkGraph, spec, task

from aiida_epw.tools.workchain import find_related_calculation
from aiida_epw.workflows.base import EpwBaseWorkChain
from aiida_epw.workflows.supercon import SuperConWorkChain

BuilderNamespace = Union[ProcessBuilderNamespace, ProcessBuilder]

__all__ = (
    "build_supercon_workgraph_from_builder",
    "build_supercon_workgraph_from_protocol",
)


def _namespace_to_dict(namespace: BuilderNamespace) -> dict[str, Any]:
    """Convert a builder namespace into a plain nested mapping."""
    return namespace._inputs(prune=True)


def _sorted_interpolation_distances(
    interpolation_distance: orm.Float | orm.List,
) -> list[orm.Float]:
    """Normalize the interpolation distance input into an ascending list of floats."""
    if isinstance(interpolation_distance, orm.Float):
        values = [interpolation_distance.value]
    else:
        values = [float(value) for value in interpolation_distance.get_list()]

    if not values:
        raise ValueError("At least one interpolation distance must be provided.")

    return [orm.Float(value) for value in sorted(values)]


@task.calcfunction()
def update_inputepw_degaussq(parameters, degaussq):
    """Return a new parameters node with an updated ``INPUTEPW.degaussq`` value."""
    updated = parameters.get_dict()
    updated.setdefault("INPUTEPW", {})["degaussq"] = degaussq.value
    return orm.Dict(updated)


@task.calcfunction()
def derive_degaussq_from_a2f(a2f):
    """Derive the smearing used by the original workflow from the highest A2F frequency."""
    frequency = a2f.get_frequency()
    return orm.Float(frequency[-1] / 100)


@task.calcfunction()
def extract_allen_dynes_tc(output_parameters):
    """Extract the Allen-Dynes critical temperature from parsed output parameters."""
    return orm.Float(output_parameters.get_dict()["Allen_Dynes_Tc"])


@task.calcfunction()
def has_converged(previous_tc, current_tc, threshold, total_runs):
    """Evaluate the same convergence condition as the original workchain."""
    if total_runs.value < 3:
        return orm.Bool(False)

    relative_change = abs(previous_tc.value - current_tc.value) / current_tc.value
    return orm.Bool(relative_change < threshold.value)


@task.calcfunction()
def should_run_final_epw(is_converged, always_run_final):
    """Decide whether the final isotropic and anisotropic runs should be launched."""
    return orm.Bool(is_converged.value or always_run_final.value)


@task.calcfunction(outputs=spec.namespace(kfpoints=Any, qfpoints=Any))
def extract_restart_meshes(parent_folder_epw):
    """Extract the fine k/q meshes from the EPW calculation used as restart parent."""
    restart_calculation = find_related_calculation(parent_folder_epw)
    return {
        "kfpoints": restart_calculation.inputs.kfpoints,
        "qfpoints": restart_calculation.inputs.qfpoints,
    }


def build_supercon_workgraph_from_builder(
    builder: ProcessBuilder,
    *,
    name: str = "supercon_workgraph",
) -> WorkGraph:
    """Build a WorkGraph from a ``SuperConWorkChain`` builder."""
    process_class = getattr(builder, "process_class", None)
    if process_class is not SuperConWorkChain:
        raise TypeError(
            "The builder must be created from `SuperConWorkChain.get_builder()` or "
            "`SuperConWorkChain.get_builder_from_protocol()`."
        )

    interp_inputs = _namespace_to_dict(builder.epw_interp)
    final_iso_inputs = _namespace_to_dict(builder.epw_final_iso)
    final_aniso_inputs = _namespace_to_dict(builder.epw_final_aniso)
    interpolation_distances = _sorted_interpolation_distances(
        builder.interpolation_distance
    )

    with WorkGraph(
        name=name,
        outputs=spec.namespace(
            converged=Any,
            epw_final_a2f_output_parameters=Any,
            epw_final_a2f_a2f=Any,
        ),
    ) as wg:
        interpolation_tasks = []
        allen_dynes_tasks = []
        updated_interp_parameters = None
        degaussq_task = None
        previous_interpolation_task = None

        for index, distance in enumerate(interpolation_distances, start=1):
            interpolation_task = wg.add_task(
                EpwBaseWorkChain,
                name=f"epw_interp_{index:02d}",
            )
            interpolation_task.set_inputs(interp_inputs)
            interpolation_task.set_inputs(
                {
                    "structure": builder.structure,
                    "parent_folder_epw": builder.parent_folder_epw,
                    "kfpoints_factor": builder.kfpoints_factor,
                    "qfpoints_distance": distance,
                }
            )

            if updated_interp_parameters is not None:
                interpolation_task.set_inputs(
                    {"parameters": updated_interp_parameters.outputs.result}
                )

            if previous_interpolation_task is not None:
                interpolation_task.waiting_on.add(previous_interpolation_task)

            interpolation_tasks.append(interpolation_task)
            previous_interpolation_task = interpolation_task

            allen_dynes_task = wg.add_task(
                extract_allen_dynes_tc,
                name=f"allen_dynes_{index:02d}",
                output_parameters=interpolation_task.outputs.output_parameters,
            )
            allen_dynes_tasks.append(allen_dynes_task)

            if index == 1:
                degaussq_task = wg.add_task(
                    derive_degaussq_from_a2f,
                    name="derive_degaussq",
                    a2f=interpolation_task.outputs.a2f,
                )
                updated_interp_parameters = wg.add_task(
                    update_inputepw_degaussq,
                    name="update_interp_degaussq",
                    parameters=interp_inputs["parameters"],
                    degaussq=degaussq_task.outputs.result,
                )

        if "convergence_threshold" in builder and len(allen_dynes_tasks) >= 2:
            converged_source = wg.add_task(
                has_converged,
                name="check_convergence",
                previous_tc=allen_dynes_tasks[-2].outputs.result,
                current_tc=allen_dynes_tasks[-1].outputs.result,
                threshold=builder.convergence_threshold,
                total_runs=orm.Int(len(interpolation_tasks)),
            ).outputs.result
        elif "convergence_threshold" in builder:
            converged_source = orm.Bool(False)
        else:
            converged_source = orm.Bool(True)

        run_final_task = wg.add_task(
            should_run_final_epw,
            name="should_run_final",
            is_converged=converged_source,
            always_run_final=builder.always_run_final,
        )

        last_interpolation_task = interpolation_tasks[-1]

        with If(run_final_task.outputs.result) as final_zone:
            restart_meshes = final_zone.add_task(
                extract_restart_meshes,
                name="extract_restart_meshes",
                parent_folder_epw=last_interpolation_task.outputs.remote_folder,
            )

            final_iso_parameters = None
            if degaussq_task is not None:
                final_iso_parameters = final_zone.add_task(
                    update_inputepw_degaussq,
                    name="update_final_iso_degaussq",
                    parameters=final_iso_inputs["parameters"],
                    degaussq=degaussq_task.outputs.result,
                )

            final_iso_task = final_zone.add_task(EpwBaseWorkChain, name="epw_final_iso")
            final_iso_task.set_inputs(final_iso_inputs)
            final_iso_overrides = {
                "structure": builder.structure,
                "parent_folder_epw": last_interpolation_task.outputs.remote_folder,
                "kfpoints": restart_meshes.outputs.kfpoints,
                "qfpoints": restart_meshes.outputs.qfpoints,
            }
            if final_iso_parameters is not None:
                final_iso_overrides["parameters"] = final_iso_parameters.outputs.result
            final_iso_task.set_inputs(final_iso_overrides)

            final_aniso_task = final_zone.add_task(
                EpwBaseWorkChain,
                name="epw_final_aniso",
            )
            final_aniso_task.set_inputs(final_aniso_inputs)
            final_aniso_task.set_inputs(
                {
                    "structure": builder.structure,
                    "parent_folder_epw": last_interpolation_task.outputs.remote_folder,
                    "kfpoints": restart_meshes.outputs.kfpoints,
                    "qfpoints": restart_meshes.outputs.qfpoints,
                }
            )
            final_aniso_task.waiting_on.add(final_iso_task)

        wg.outputs.converged = converged_source
        wg.outputs.epw_final_a2f_output_parameters = (
            last_interpolation_task.outputs.output_parameters
        )
        wg.outputs.epw_final_a2f_a2f = last_interpolation_task.outputs.a2f

    return wg


def build_supercon_workgraph_from_protocol(
    epw_code,
    parent_epw,
    *,
    protocol=None,
    overrides=None,
    scon_epw_code=None,
    parent_folder_epw=None,
    name: str = "supercon_workgraph",
    **kwargs,
) -> WorkGraph:
    """Build the superconductivity WorkGraph from the existing protocol helper."""
    builder = SuperConWorkChain.get_builder_from_protocol(
        epw_code=epw_code,
        parent_epw=parent_epw,
        protocol=protocol,
        overrides=overrides,
        scon_epw_code=scon_epw_code,
        parent_folder_epw=parent_folder_epw,
        **kwargs,
    )

    return build_supercon_workgraph_from_builder(builder, name=name)
