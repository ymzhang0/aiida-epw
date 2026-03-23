"""WorkGraph implementation of the superconductivity workflow."""

from __future__ import annotations

from typing import Any, Union

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ProcessBuilder
from aiida.engine.processes.builder import ProcessBuilderNamespace
from aiida_workgraph import If, WorkGraph, spec, task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_epw.tools.workchain import find_related_calculation
from aiida_epw.workflows.base import EpwBaseWorkChain
from aiida_epw.workflows.supercon import SuperConWorkChain

BuilderNamespace = Union[ProcessBuilderNamespace, ProcessBuilder]

__all__ = (
    "supercon",
)


def get_protocol_inputs(
    protocol: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """Return the inputs for the EPW preparation workflow based on a protocol."""
    from importlib_resources import files
    from aiida_epw.workflows import protocols
    from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

    # 1. Load protocol file from workflows side
    filepath = files(protocols) / "supercon.yaml"

    AdHocProtocol = type("AdHocProtocol", (ProtocolMixin,), {
        "get_protocol_filepath": classmethod(lambda cls: filepath),
        "_validate_override_keys": classmethod(lambda cls, overrides: None)
    })

    return AdHocProtocol.get_protocol_inputs(protocol, overrides)


def _namespace_to_dict(namespace: BuilderNamespace) -> dict[str, Any]:
    """Convert a builder namespace into a plain nested mapping."""
    return get_dict_from_builder(namespace)


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


def _get_prep_reciprocal_points(parent_prep):
    """Return the reciprocal meshes produced by the prep workgraph."""
    reciprocal_points = (
        parent_prep.base.links.get_outgoing(link_label_filter="generate_reciprocal_points")
        .first()
        .node
    )
    return AttributeDict(
        {
            "kpoints": reciprocal_points.outputs.kpoints_nscf,
            "qpoints": reciprocal_points.outputs.qpoints,
        }
    )


@task(outputs=spec.namespace(
    converged=Any,
    epw_final_a2f_output_parameters=Any,
    epw_final_a2f_a2f=Any,
))
def results(converged, output_parameters, a2f):
    """Expose the final outputs following the original ``results`` step."""
    return {
        "converged": converged,
        "epw_final_a2f_output_parameters": output_parameters,
        "epw_final_a2f_a2f": a2f,
    }


@task.graph(outputs=["converged", "epw_final_a2f_output_parameters", "epw_final_a2f_a2f"])
def supercon(
    code,
    parent_epw,
    protocol=None,
    overrides=None,
    parent_folder_epw=None,
    **kwargs,
):
    """Superconductivity WorkGraph."""
    if overrides and hasattr(overrides, "items"):
        overrides = dict(overrides.items())

    inputs = get_protocol_inputs(protocol, overrides)

    if parent_epw.process_label == "WorkGraph<prep>":
        reciprocal_points = _get_prep_reciprocal_points(parent_epw)
        epw_source = AttributeDict(
            {
                "inputs": AttributeDict(
                    {
                        "structure": parent_epw.inputs.structure,
                        "kpoints": reciprocal_points.kpoints,
                        "qpoints": reciprocal_points.qpoints,
                    }
                )
            }
        )
    elif parent_epw.process_label == "EpwPrepWorkChain":
        epw_source = (
            parent_epw.base.links.get_outgoing(link_label_filter="epw_base")
            .first()
            .node
        )
    elif parent_epw.process_label == "EpwBaseWorkChain":
        epw_source = parent_epw
    else:
        raise ValueError(f"Invalid parent_epw process: {parent_epw.process_label}")

    if parent_folder_epw is None:
        if parent_epw.process_label == "WorkGraph<prep>":
            parent_folder_epw = parent_epw.outputs.epw_stash
        else:
            from aiida_epw.workflows.supercon import get_restart_parent_folder

            parent_folder_epw = get_restart_parent_folder(parent_epw)

    sub_inputs = {}
    for epw_namespace in (
        "epw_interp",
        "epw_final_iso",
        "epw_final_aniso",
    ):
        epw_inputs = inputs.get(epw_namespace, {})
        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=code,
            structure=epw_source.inputs.structure,
            protocol=protocol,
            overrides=epw_inputs,
            **kwargs,
        )
        epw_builder.kpoints = epw_source.inputs.kpoints
        epw_builder.qpoints = epw_source.inputs.qpoints
        if "settings" in epw_inputs:
            epw_builder.settings = orm.Dict(epw_inputs["settings"])
            
        sub_inputs[epw_namespace] = get_dict_from_builder(epw_builder)

    interp_inputs = sub_inputs["epw_interp"]
    final_iso_inputs = sub_inputs["epw_final_iso"]
    final_aniso_inputs = sub_inputs["epw_final_aniso"]

    distance_input = inputs["interpolation_distance"]
    if isinstance(distance_input, float):
         distance_node = orm.Float(distance_input)
    elif isinstance(distance_input, list):
         distance_node = orm.List(distance_input)
    else:
         distance_node = distance_input
         
    interpolation_distances = _sorted_interpolation_distances(distance_node)

    convergence_threshold = inputs.get("convergence_threshold")
    convergence_threshold_node = orm.Float(convergence_threshold) if convergence_threshold is not None else None
    always_run_final = orm.Bool(inputs.get("always_run_final", False))
    structure = epw_source.inputs.structure
    kfpoints_factor = orm.Int(inputs.get("kfpoints_factor", 1))

    interpolation_tasks = []
    allen_dynes_tasks = []
    updated_interp_parameters = None
    degaussq_task = None
    previous_interpolation_task = None

    for index, distance in enumerate(interpolation_distances, start=1):
        EpwInterpTask = task(identifier=f"epw_interp_{index:02d}")(EpwBaseWorkChain)
        
        inputs_merged = dict(interp_inputs)
        inputs_merged.update({
            "structure": structure,
            "parent_folder_epw": parent_folder_epw,
            "kfpoints_factor": kfpoints_factor,
            "qfpoints_distance": distance,
        })
        if updated_interp_parameters is not None:
             inputs_merged["parameters"] = updated_interp_parameters.result

        interpolation_task = EpwInterpTask(**inputs_merged)

        if previous_interpolation_task is not None:
            interpolation_task._task.waiting_on.add(previous_interpolation_task._task)

        interpolation_tasks.append(interpolation_task)
        previous_interpolation_task = interpolation_task

        allen_dynes_task = extract_allen_dynes_tc(
            output_parameters=interpolation_task.output_parameters,
        )
        allen_dynes_tasks.append(allen_dynes_task)

        if index == 1:
            degaussq_task = derive_degaussq_from_a2f(
                a2f=interpolation_task.a2f,
            )
            updated_interp_parameters = update_inputepw_degaussq(
                parameters=interp_inputs["parameters"],
                degaussq=degaussq_task.result,
            )

    if convergence_threshold_node is not None and len(allen_dynes_tasks) >= 2:
        converged_source = has_converged(
            previous_tc=allen_dynes_tasks[-2].result,
            current_tc=allen_dynes_tasks[-1].result,
            threshold=convergence_threshold_node,
            total_runs=orm.Int(len(interpolation_tasks)),
        ).result
    elif convergence_threshold_node is not None:
        converged_source = orm.Bool(False)
    else:
        converged_source = orm.Bool(True)

    run_final_task = should_run_final_epw(
        is_converged=converged_source,
        always_run_final=always_run_final,
    )

    last_interpolation_task = interpolation_tasks[-1]

    with If(run_final_task.result):
        restart_meshes = extract_restart_meshes(
            parent_folder_epw=last_interpolation_task.remote_folder,
        )

        final_iso_parameters = None
        if degaussq_task is not None:
            final_iso_parameters = update_inputepw_degaussq(
                parameters=final_iso_inputs["parameters"],
                degaussq=degaussq_task.result,
            )

        EpwFinalIsoTask = task(identifier="epw_final_iso")(EpwBaseWorkChain)
        final_iso_inputs_merged = dict(final_iso_inputs)
        final_iso_inputs_merged.update({
             "structure": structure,
             "parent_folder_epw": last_interpolation_task.remote_folder,
             "kfpoints": restart_meshes.kfpoints,
             "qfpoints": restart_meshes.qfpoints,
        })
        if final_iso_parameters is not None:
             final_iso_inputs_merged["parameters"] = final_iso_parameters.result

        final_iso_task = EpwFinalIsoTask(**final_iso_inputs_merged)

        EpwFinalAnisoTask = task(identifier="epw_final_aniso")(EpwBaseWorkChain)
        final_aniso_inputs_merged = dict(final_aniso_inputs)
        final_aniso_inputs_merged.update({
             "structure": structure,
             "parent_folder_epw": last_interpolation_task.remote_folder,
             "kfpoints": restart_meshes.kfpoints,
             "qfpoints": restart_meshes.qfpoints,
        })
        final_aniso_task = EpwFinalAnisoTask(**final_aniso_inputs_merged)
        final_aniso_task._task.waiting_on.add(final_iso_task._task)

    return results(
        converged=converged_source,
        output_parameters=last_interpolation_task.output_parameters,
        a2f=last_interpolation_task.a2f,
    )
