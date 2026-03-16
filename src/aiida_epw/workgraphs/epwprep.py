"""WorkGraph implementation of the EPW preparation workflow."""

from __future__ import annotations

from typing import Any, Union

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida.engine.processes.builder import ProcessBuilderNamespace
from aiida_workgraph import WorkGraph, spec, task
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90OptimizeWorkChain,
)

from aiida_epw.tools.workchain import get_parent_folder_calculation
from aiida_epw.workflows.base import EpwBaseWorkChain
from aiida_epw.workflows.prep import (
    EpwPrepWorkChain,
    should_run_bands_interpolation,
    validate_inputs,
)

BuilderNamespace = Union[ProcessBuilderNamespace, ProcessBuilder]

__all__ = (
    "build_epw_prep_workgraph_from_builder",
    "build_epw_prep_workgraph_from_protocol",
)


def _create_kpoints_from_distance_node(
    structure: orm.StructureData,
    distance: orm.Float,
    force_parity: orm.Bool,
) -> orm.KpointsData:
    """Create a k-point mesh with the same logic as the QE helper calcfunction."""
    from numpy import linalg

    epsilon = 1e-5

    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(
        distance.value, force_parity=force_parity.value
    )

    lengths_vector = [linalg.norm(vector) for vector in structure.cell]
    lengths_kpoint = kpoints.get_kpoints_mesh()[0]

    is_symmetric_cell = all(
        abs(length - lengths_vector[0]) < epsilon for length in lengths_vector
    )
    is_symmetric_mesh = all(length == lengths_kpoint[0] for length in lengths_kpoint)

    if is_symmetric_cell and not is_symmetric_mesh:
        nkpoints = max(lengths_kpoint)
        kpoints.set_kpoints_mesh([nkpoints if pbc else 1 for pbc in structure.pbc])

    return kpoints


def _namespace_to_dict(namespace: BuilderNamespace) -> dict[str, Any]:
    """Convert a builder namespace into a plain nested mapping."""
    return namespace._inputs(prune=True)


def _namespace_get(
    namespace: BuilderNamespace | dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Get an input from a builder namespace or plain mapping."""
    if isinstance(namespace, dict):
        return namespace.get(key, default)

    return namespace[key] if key in namespace else default


def _validation_inputs_from_builder(builder: ProcessBuilder) -> dict[str, Any]:
    """Build the minimal input mapping required by the original validator."""
    inputs = {
        "do_bands_interpolation": _namespace_get(
            builder, "do_bands_interpolation", orm.Bool(True)
        ),
    }

    if "w90_bands" in builder:
        inputs["w90_bands"] = {}

    if "epw_bands" in builder:
        inputs["epw_bands"] = {}

    return inputs


def _get_wannier_workchain_class(
    w90_inputs: dict[str, Any],
) -> type[Wannier90BandsWorkChain] | type[Wannier90OptimizeWorkChain]:
    """Return the concrete Wannier workchain class for the provided inputs."""
    if "reference_bands" in w90_inputs:
        return Wannier90OptimizeWorkChain

    return Wannier90BandsWorkChain


def _is_optimize_disproj_enabled(
    w90_class: type[Wannier90BandsWorkChain] | type[Wannier90OptimizeWorkChain],
    w90_inputs: dict[str, Any],
) -> bool:
    """Return whether the optimal Wannier checkpoint folder should be used."""
    optimize_disproj = _namespace_get(w90_inputs, "optimize_disproj", orm.Bool(False))

    if isinstance(optimize_disproj, orm.Bool):
        optimize_disproj = optimize_disproj.value

    return w90_class is Wannier90OptimizeWorkChain and bool(optimize_disproj)


@task.calcfunction(
    outputs=spec.namespace(
        qpoints=Any,
        kpoints_scf=Any,
        kpoints_nscf_mesh=Any,
        kpoints_nscf_explicit=Any,
    )
)
def generate_reciprocal_points(
    structure,
    qpoints_distance,
    kpoints_distance_scf,
    kpoints_factor_nscf,
    force_parity,
):
    """Generate the reciprocal-space meshes required by the EPW preparation graph."""
    qpoints = _create_kpoints_from_distance_node(
        structure, qpoints_distance, force_parity
    )
    kpoints_scf = _create_kpoints_from_distance_node(
        structure, kpoints_distance_scf, force_parity
    )

    qpoints_mesh = qpoints.get_kpoints_mesh()[0]
    kpoints_nscf_mesh = orm.KpointsData()
    kpoints_nscf_mesh.set_kpoints_mesh(
        [value * kpoints_factor_nscf.value for value in qpoints_mesh]
    )

    return {
        "qpoints": qpoints,
        "kpoints_scf": kpoints_scf,
        "kpoints_nscf_mesh": kpoints_nscf_mesh,
        "kpoints_nscf_explicit": get_explicit_kpoints(kpoints_nscf_mesh),
    }


@task.calcfunction(outputs=spec.namespace(parameters=Any))
def update_wannier90_parameters(parameters, kpoints_mesh):
    """Update the Wannier90 ``mp_grid`` setting to match the NSCF mesh."""
    updated = parameters.get_dict()
    updated["mp_grid"] = kpoints_mesh.get_kpoints_mesh()[0]
    return {"parameters": orm.Dict(updated)}


@task.calcfunction()
def create_gamma_mesh():
    """Create the gamma-only mesh used by the transformation EPW run."""
    fine_points = orm.KpointsData()
    fine_points.set_kpoints_mesh([1, 1, 1])
    return fine_points


@task.calcfunction()
def extract_kpoints_from_bands(band_structure):
    """Convert a ``BandsData`` output into a standalone ``KpointsData`` node."""
    kpoints = orm.KpointsData()
    kpoints.set_cell(band_structure.cell, band_structure.pbc)

    points, weights = band_structure.get_kpoints(also_weights=True)
    labels = band_structure.labels if band_structure.labels else None
    kpoints.set_kpoints(points, weights=weights, labels=labels)

    return kpoints


def build_epw_prep_workgraph_from_builder(
    builder: ProcessBuilder,
    *,
    name: str = "epw_prep_workgraph",
) -> WorkGraph:
    """Build a WorkGraph from an ``EpwPrepWorkChain`` builder."""
    process_class = getattr(builder, "process_class", None)
    if process_class is not EpwPrepWorkChain:
        raise TypeError(
            "The builder must be created from `EpwPrepWorkChain.get_builder()` or "
            "`EpwPrepWorkChain.get_builder_from_protocol()`."
        )

    validation_error = validate_inputs(_validation_inputs_from_builder(builder))
    if validation_error is not None:
        raise ValueError(validation_error)

    w90_inputs = _namespace_to_dict(builder.w90_bands)
    w90_class = _get_wannier_workchain_class(w90_inputs)

    ph_inputs = _namespace_to_dict(builder.ph_base)
    epw_base_inputs = _namespace_to_dict(builder.epw_base)
    do_bands_interpolation = should_run_bands_interpolation(
        _validation_inputs_from_builder(builder)
    )
    epw_bands_inputs = (
        _namespace_to_dict(builder.epw_bands) if do_bands_interpolation else None
    )

    force_parity = _namespace_get(builder, "kpoints_force_parity", orm.Bool(False))

    wg = WorkGraph(
        name=name,
        outputs=spec.namespace(
            retrieved=Any,
            epw_folder=Any,
        ),
    )

    reciprocal_points = wg.add_task(
        generate_reciprocal_points,
        name="generate_reciprocal_points",
        structure=builder.structure,
        qpoints_distance=builder.qpoints_distance,
        kpoints_distance_scf=builder.kpoints_distance_scf,
        kpoints_factor_nscf=builder.kpoints_factor_nscf,
        force_parity=force_parity,
    )

    wannier90_parameters = wg.add_task(
        update_wannier90_parameters,
        name="update_wannier90_parameters",
        parameters=w90_inputs["wannier90"]["wannier90"]["parameters"],
        kpoints_mesh=reciprocal_points.outputs.kpoints_nscf_mesh,
    )

    w90_task = wg.add_task(w90_class, name="w90_bands")
    w90_task.set_inputs(w90_inputs)
    w90_task.set_inputs(
        {
            "structure": builder.structure,
            "scf.kpoints": reciprocal_points.outputs.kpoints_scf,
            "nscf.kpoints": reciprocal_points.outputs.kpoints_nscf_explicit,
            "wannier90.wannier90.kpoints": reciprocal_points.outputs.kpoints_nscf_explicit,
            "wannier90.wannier90.parameters": wannier90_parameters.outputs.parameters,
        }
    )

    ph_task = wg.add_task(PhBaseWorkChain, name="ph_base")
    ph_task.set_inputs(ph_inputs)

    parent_folder_ph = _namespace_get(builder, "parent_folder_ph")
    if parent_folder_ph is not None:
        parent_calculation = get_parent_folder_calculation(parent_folder_ph)
        if parent_calculation.process_label == "PhCalculation":
            ph_task.set_inputs(
                {
                    "ph.parent_folder": parent_folder_ph,
                    "ph.qpoints": parent_calculation.inputs.qpoints,
                }
            )
        else:
            ph_task.set_inputs(
                {"ph.parent_folder": w90_task.outputs.scf.remote_folder}
            )
    else:
        ph_task.set_inputs({"ph.parent_folder": w90_task.outputs.scf.remote_folder})

    ph_task.set_inputs({"qpoints": reciprocal_points.outputs.qpoints})

    gamma_mesh = wg.add_task(create_gamma_mesh, name="create_gamma_mesh")

    epw_base_task = wg.add_task(EpwBaseWorkChain, name="epw_base")
    epw_base_task.set_inputs(epw_base_inputs)

    if _is_optimize_disproj_enabled(w90_class, w90_inputs):
        parent_folder_chk = w90_task.outputs.wannier90_optimal.remote_folder
    else:
        parent_folder_chk = w90_task.outputs.wannier90.remote_folder

    epw_base_task.set_inputs(
        {
            "structure": builder.structure,
            "parent_folder_ph": ph_task.outputs.remote_folder,
            "parent_folder_nscf": w90_task.outputs.nscf.remote_folder,
            "parent_folder_chk": parent_folder_chk,
            "kpoints": reciprocal_points.outputs.kpoints_nscf_mesh,
            "kfpoints": gamma_mesh.outputs.result,
            "qpoints": reciprocal_points.outputs.qpoints,
            "qfpoints": gamma_mesh.outputs.result,
        }
    )

    if do_bands_interpolation and epw_bands_inputs is not None:
        if "bands_kpoints" in w90_inputs:
            bands_kpoints = w90_inputs["bands_kpoints"]
        else:
            bands_kpoints_task = wg.add_task(
                extract_kpoints_from_bands,
                name="extract_bands_kpoints",
                band_structure=w90_task.outputs.band_structure,
            )
            bands_kpoints = bands_kpoints_task.outputs.result

        epw_bands_task = wg.add_task(EpwBaseWorkChain, name="epw_bands")
        epw_bands_task.set_inputs(epw_bands_inputs)
        epw_bands_task.set_inputs(
            {
                "structure": builder.structure,
                "parent_folder_epw": epw_base_task.outputs.remote_stash,
                "kpoints": reciprocal_points.outputs.kpoints_nscf_mesh,
                "qpoints": reciprocal_points.outputs.qpoints,
                "qfpoints": bands_kpoints,
                "kfpoints": bands_kpoints,
            }
        )

    wg.outputs.retrieved = epw_base_task.outputs.retrieved
    wg.outputs.epw_folder = epw_base_task.outputs.remote_stash

    return wg


def build_epw_prep_workgraph_from_protocol(
    codes,
    structure,
    *,
    protocol=None,
    overrides=None,
    wannier_projection_type=None,
    reference_bands=None,
    bands_kpoints=None,
    parent_folder_ph=None,
    name: str = "epw_prep_workgraph",
    **kwargs,
) -> WorkGraph:
    """Build the EPW preparation WorkGraph from the existing protocol helper."""
    if wannier_projection_type is None:
        from aiida_wannier90_workflows.common.types import WannierProjectionType

        wannier_projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE

    builder = EpwPrepWorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        wannier_projection_type=wannier_projection_type,
        reference_bands=reference_bands,
        bands_kpoints=bands_kpoints,
        parent_folder_ph=parent_folder_ph,
        **kwargs,
    )

    return build_epw_prep_workgraph_from_builder(builder, name=name)
