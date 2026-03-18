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


@task.calcfunction()
def results(converged, output_parameters, a2f):
    """Expose the final outputs following the original ``results`` step."""
    return {
        "converged": converged,
        "epw_final_a2f_output_parameters": output_parameters,
        "epw_final_a2f_a2f": a2f,
    }


@task.graph(outputs=spec.namespace(
    converged=Any, 
    remote_folder=Any, 
    output_parameters=Any, 
    a2f=Any,
    degaussq=Any
))
def run_convergence(interp_inputs, interpolation_distances, convergence_threshold_node, structure, parent_folder_epw, kfpoints_factor):
    """Run the interpolation EpwBaseWorkChain convergence loop."""
    from aiida_epw.workflows.base import EpwBaseWorkChain
    
    interpolation_tasks = []
    allen_dynes_tasks = []
    updated_interp_parameters = None
    degaussq_task_node = None
    previous_task = None

    for index, distance in enumerate(interpolation_distances, start=1):
         EpwInterpTask = task(identifier=f"epw_interp_{index:02d}")(EpwBaseWorkChain)
         
         inputs = dict(interp_inputs)
         inputs.update({
             "structure": structure,
             "parent_folder_epw": parent_folder_epw,
             "kfpoints_factor": kfpoints_factor,
             "qfpoints_distance": distance,
         })
         if updated_interp_parameters is not None:
              inputs["parameters"] = updated_interp_parameters.result
              
         t = EpwInterpTask(**inputs)
         interpolation_tasks.append(t)
         
         if previous_task:
              t.waiting_on.add(previous_task)
         previous_task = t

         tc_t = extract_allen_dynes_tc(output_parameters=t.outputs.output_parameters)
         allen_dynes_tasks.append(tc_t)

         if index == 1:
              degaussq_task_node = derive_degaussq_from_a2f(a2f=t.outputs.a2f)
              updated_interp_parameters = update_inputepw_degaussq(
                  parameters=interp_inputs["parameters"], 
                  degaussq=degaussq_task_node.result
              )

    if convergence_threshold_node is not None and len(allen_dynes_tasks) >= 2:
         converged = has_converged(
             previous_tc=allen_dynes_tasks[-2].result, 
             current_tc=allen_dynes_tasks[-1].result, 
             threshold=convergence_threshold_node, 
             total_runs=orm.Int(len(interpolation_tasks))
         ).result
    elif convergence_threshold_node is not None:
         converged = orm.Bool(False)
    else:
         converged = orm.Bool(True)

    last_t = interpolation_tasks[-1]
    return {
        "converged": converged,
        "remote_folder": last_t.outputs.remote_folder,
        "output_parameters": last_t.outputs.output_parameters,
        "a2f": last_t.outputs.a2f,
        "degaussq": degaussq_task_node.result if degaussq_task_node else None,
    }


@task.graph(outputs=spec.namespace(epw_folder=Any))
def run_final_epw_iso(final_iso_inputs, structure, parent_folder_epw, kfpoints, qfpoints, degaussq):
    """Run final isotropic EPW task graph."""
    from aiida_epw.workflows.base import EpwBaseWorkChain
    EpwFinalIsoTask = task(identifier="epw_final_iso")(EpwBaseWorkChain)
    
    inputs = dict(final_iso_inputs)
    inputs.update({
         "structure": structure,
         "parent_folder_epw": parent_folder_epw,
         "kfpoints": kfpoints,
         "qfpoints": qfpoints,
    })
    
    updated = update_inputepw_degaussq(
         parameters=final_iso_inputs["parameters"], 
         degaussq=degaussq
    )
    inputs["parameters"] = updated.result
         
    final_iso = EpwFinalIsoTask(**inputs)
    return {"epw_folder": final_iso.outputs.remote_folder}


@task.graph(outputs=spec.namespace(epw_folder=Any))
def run_final_epw_aniso(final_aniso_inputs, structure, parent_folder_epw, kfpoints, qfpoints, final_iso_folder):
    """Run final anisotropic EPW task graph."""
    from aiida_epw.workflows.base import EpwBaseWorkChain
    EpwFinalAnisoTask = task(identifier="epw_final_aniso")(EpwBaseWorkChain)
    
    inputs = dict(final_aniso_inputs)
    inputs.update({
         "structure": structure,
         "parent_folder_epw": parent_folder_epw,
         "kfpoints": kfpoints,
         "qfpoints": qfpoints,
    })
    final_aniso = EpwFinalAnisoTask(**inputs)
    final_aniso.waiting_on.add(final_iso_folder)
    return {"epw_folder": final_aniso.outputs.remote_folder}


@task.graph(outputs=["converged", "epw_final_a2f_output_parameters", "epw_final_a2f_a2f"])
def supercon(
    epw_code,
    parent_epw,
    protocol=None,
    overrides=None,
    scon_epw_code=None,
    parent_folder_epw=None,
    **kwargs,
):
    """Superconductivity WorkGraph."""
    inputs = get_protocol_inputs(protocol, overrides)

    if parent_epw.process_label == "EpwPrepWorkChain":
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
        from aiida_epw.workflows.supercon import get_restart_parent_folder
        parent_folder_epw = get_restart_parent_folder(parent_epw)

    sub_inputs = {}
    from aiida_epw.workflows.base import EpwBaseWorkChain
    for epw_namespace in ("epw_interp", "epw_final_iso", "epw_final_aniso"):
        epw_inputs = inputs.get(epw_namespace, {})
        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=scon_epw_code if (epw_namespace == "epw_interp" and scon_epw_code is not None) else epw_code,
            structure=epw_source.inputs.structure,
            protocol=protocol,
            overrides=epw_inputs,
            **kwargs,
        )
        epw_builder.kpoints = epw_source.inputs.kpoints
        epw_builder.qpoints = epw_source.inputs.qpoints
        if "settings" in epw_inputs:
            epw_builder.settings = orm.Dict(epw_inputs["settings"])
        sub_inputs[epw_namespace] = epw_builder._inputs(prune=False)

    distance_input = inputs["interpolation_distance"]
    interpolation_distances = _sorted_interpolation_distances(distance_input if isinstance(distance_input, (orm.Float, orm.List)) else orm.Float(distance_input))

    convergence_threshold = inputs.get("convergence_threshold")
    convergence_threshold_node = orm.Float(convergence_threshold) if convergence_threshold is not None else None
    always_run_final = orm.Bool(inputs.get("always_run_final", False))
    structure = epw_source.inputs.structure
    kfpoints_factor = orm.Int(inputs.get("kfpoints_factor", 1))

    # --- 1. Run Convergence Loop (Nested) ---
    conv = run_convergence(
        interp_inputs=sub_inputs["epw_interp"],
        interpolation_distances=interpolation_distances,
        convergence_threshold_node=convergence_threshold_node,
        structure=structure,
        parent_folder_epw=parent_folder_epw,
        kfpoints_factor=kfpoints_factor
    )

    run_final_task = should_run_final_epw(
        is_converged=conv.converged,
        always_run_final=always_run_final
    )

    with If(run_final_task.result):
        restart_meshes = extract_restart_meshes(parent_folder_epw=conv.remote_folder)
        
        final_iso = run_final_epw_iso(
            final_iso_inputs=sub_inputs["epw_final_iso"],
            structure=structure,
            parent_folder_epw=conv.remote_folder,
            kfpoints=restart_meshes.outputs.kfpoints,
            qfpoints=restart_meshes.outputs.qfpoints,
            degaussq=conv.degaussq
        )
        
        final_aniso = run_final_epw_aniso(
            final_aniso_inputs=sub_inputs["epw_final_aniso"],
            structure=structure,
            parent_folder_epw=conv.remote_folder,
            kfpoints=restart_meshes.outputs.kfpoints,
            qfpoints=restart_meshes.outputs.qfpoints,
            final_iso_folder=final_iso.epw_folder
        )

    return results(
        converged=conv.converged,
        output_parameters=conv.output_parameters,
        a2f=conv.a2f
    )
