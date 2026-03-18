"""WorkGraph implementation of the EPW preparation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiida import orm
from aiida.engine import ProcessBuilder, ExitCode
from aiida_workgraph import If, spec, task
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints
from aiida_workgraph.utils import get_dict_from_builder
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90OptimizeWorkChain,
)
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_epw.tools.workchain import get_parent_folder_calculation, get_target_basepath
from aiida_epw.workflows.base import EpwBaseWorkChain

Wannier90BandsTask = task(Wannier90BandsWorkChain)
Wannier90OptimizeTask = task(Wannier90OptimizeWorkChain)
PhBaseTask = task(PhBaseWorkChain)
EpwBaseTask = task(EpwBaseWorkChain)

__all__ = (
    "prep",
)

def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the `EpwPrepWorkChain`."""
    do_bands = inputs.get("do_bands_interpolation", True)
    if isinstance(do_bands, orm.Bool):
        do_bands = do_bands.value

    if "w90_bands" not in inputs:
        return (
            "`w90_bands` inputs are required because this work chain needs the "
            "NSCF and Wannier checkpoint folders produced by the Wannier90 step."
        )

    if do_bands and "epw_bands" not in inputs:
        return (
            "`epw_bands` inputs are required when `do_bands_interpolation` is enabled."
        )


def should_run_bands_interpolation(inputs) -> bool:
    """Return whether the EPW bands interpolation step should run."""
    do_bands = inputs.get("do_bands_interpolation", True)
    if isinstance(do_bands, orm.Bool):
        do_bands = do_bands.value

    return bool(do_bands) and "epw_bands" in inputs

@task.calcfunction(
    outputs=spec.namespace(
        kpoints_scf=Any,
        qpoints=Any,
        kpoints_nscf=Any,
    )
)
def generate_reciprocal_points(
    structure,
    force_parity,
    kpoints_distance_scf,
    qpoints_distance,
    kpoints_factor_nscf
):
    """Generate the SCF k-point mesh, Q-point mesh, and NSCF k-point mesh for the EPW parameters."""
    inputs_scf = {
        "structure": structure,
        "distance": kpoints_distance_scf,
        "force_parity": force_parity,
        "metadata": {"call_link_label": "create_kpoints_from_distance"},
    }
    inputs_q = {
        "structure": structure,
        "distance": qpoints_distance,
        "force_parity": force_parity,
        "metadata": {"call_link_label": "create_qpoints_from_distance"},
    }
    kpoints_scf = create_kpoints_from_distance(**inputs_scf)
    qpoints = create_kpoints_from_distance(**inputs_q)
    qpoints_mesh = qpoints.get_kpoints_mesh()[0]
    kpoints_nscf = orm.KpointsData()
    kpoints_nscf.set_kpoints_mesh(
        [value * kpoints_factor_nscf.value for value in qpoints_mesh]
    )

    return {
        "kpoints_scf": kpoints_scf,
        "qpoints": qpoints,
        "kpoints_nscf": kpoints_nscf,
    }


@task()
def should_run_wannier90(w90_parameters) -> bool:
    """Mirror the outline guard for the Wannier90 branch."""
    bands_plot = False
    if w90_parameters is not None:
        bands_plot = w90_parameters.get_dict().get("bands_plot", False)
    return orm.Bool(bands_plot)


@task.calcfunction(outputs=spec.namespace(parameters=Any))
def update_wannier90_parameters(parameters, kpoints_nscf):
    """Update the Wannier90 ``mp_grid`` to match the NSCF mesh."""
    updated = parameters.get_dict()
    updated["mp_grid"] = kpoints_nscf.get_kpoints_mesh()[0]
    return {"parameters": orm.Dict(updated)}





@task.calcfunction()
def create_kpoints_gamma():
    """Create the gamma-only mesh used by the transformation EPW run."""
    gamma = orm.KpointsData()
    gamma.set_kpoints_mesh([1, 1, 1])
    return gamma


@task()
def should_run_epw_bands(do_bands_interpolation, epw_parameters) -> bool:
    """Mirror the outline guard for the EPW bands interpolation branch."""
    do_bands = False
    if do_bands_interpolation is not None:
        do_bands = do_bands_interpolation.value
        
    bands_plot = False
    if epw_parameters is not None:
        bands_plot = epw_parameters.get_dict().get("band_plot", False)
        
    return orm.Bool(bool(do_bands) and bands_plot)


@task.calcfunction()
def extract_kpoints_path(band_structure):
    """Convert a ``BandsData`` output into a standalone ``KpointsData`` node."""
    kpoints = orm.KpointsData()
    kpoints.set_cell(band_structure.cell, band_structure.pbc)

    points, weights = band_structure.get_kpoints(also_weights=True)
    labels = band_structure.labels if band_structure.labels else None
    kpoints.set_kpoints(points, weights=weights, labels=labels)

    return kpoints


@task.calcfunction(outputs=spec.namespace(bands_kpoints=Any))
def prepare_epw_bands_kpoints(band_structure):
    """Prepare the bands interpolation k-points using the original workflow priority."""
    return {"bands_kpoints": extract_kpoints_path(band_structure).result}





@task.calcfunction(outputs=spec.namespace(retrieved=Any, epw_folder=Any))
def results(retrieved, epw_folder):
    """Expose the final outputs following the original ``results`` step."""
    return {
        "retrieved": retrieved,
        "epw_folder": epw_folder,
    }


@task.graph(outputs=["retrieved", "epw_folder"])
def prep(
    codes: dict,
    structure: orm.StructureData,
    protocol: str | None = None,
    overrides: dict | None = None,
    wannier_projection_type=None,
    reference_bands: orm.BandsData | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    **kwargs,
):
    """Compose the EPW preparation outline as a WorkGraph."""
    if hasattr(codes, "items"):
        codes = dict(codes.items())
    if overrides and hasattr(overrides, "items"):
        overrides = dict(overrides.items())
        
    inputs = get_protocol_inputs(protocol, overrides)
    pseudo_family = inputs.pop("pseudo_family", None)

    if wannier_projection_type is None:
        wannier_projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE

    if reference_bands:
        w90_builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            structure=structure,
            codes=codes,
            pseudo_family=pseudo_family,
            overrides=inputs.get("w90_bands", {}),
            projection_type=wannier_projection_type,
            reference_bands=reference_bands,
            bands_kpoints=bands_kpoints,
        )
        w90_builder.separate_plotting = False
    else:
        w90_builder = Wannier90BandsWorkChain.get_builder_from_protocol(
            structure=structure,
            codes=codes,
            pseudo_family=pseudo_family,
            overrides=inputs.get("w90_bands", {}),
            projection_type=wannier_projection_type,
            bands_kpoints=bands_kpoints,
        )

    w90_bands = get_dict_from_builder(w90_builder)
    if wannier_projection_type == WannierProjectionType.ATOMIC_PROJECTORS_QE:
        w90_bands.pop("projwfc", None)

    w90_bands.pop("structure", None)
    w90_bands.pop("open_grid", None)

    # ph_base
    ph_base_inputs = inputs.get("ph_base", {})
    ph_code = codes.get("ph")
    if ph_code and "target_base" not in ph_base_inputs.get("ph", {}).get("metadata", {}).get("options", {}).get("stash", {}):
        ph_stash = ph_base_inputs.setdefault("ph", {}).setdefault("metadata", {}).setdefault("options", {}).setdefault("stash", {})
        ph_stash["target_base"] = get_target_basepath(ph_code.computer)
        ph_stash["stash_mode"] = ph_stash.get("stash_mode", "copy")

    ph_base_builder = PhBaseWorkChain.get_builder_from_protocol(
        codes["ph"], None, protocol, overrides=ph_base_inputs, **kwargs
    )
    ph_base = get_dict_from_builder(ph_base_builder)
    ph_base.pop("clean_workdir", None)
    ph_base.pop("qpoints_distance", None)

    # epw_base and optional epw_bands
    epw_base = {}
    epw_bands = {}
    epw_code = codes.get("epw")
    for namespace in ["epw_base", "epw_bands"]:
        if namespace == "epw_bands" and not inputs.get("do_bands_interpolation", True):
            continue

        epw_inputs = inputs.get(namespace, {})
        if namespace == "epw_base" and epw_code and "options" in epw_inputs:
            if "target_base" not in epw_inputs.get("options", {}).get("stash", {}):
                epw_stash = epw_inputs.setdefault("options", {}).setdefault("stash", {})
                epw_stash["target_base"] = get_target_basepath(epw_code.computer)
                epw_stash["stash_mode"] = epw_stash.get("stash_mode", "copy")

        epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
            code=codes["epw"],
            structure=structure,
            protocol=protocol,
            overrides=epw_inputs,
            **kwargs,
        )

        if "settings" in epw_inputs:
            epw_builder.settings = orm.Dict(epw_inputs["settings"])
        if "parallelization" in epw_inputs:
            epw_builder.parallelization = orm.Dict(epw_inputs["parallelization"])

        if namespace == "epw_base":
            epw_base = get_dict_from_builder(epw_builder)
        else:
            epw_bands = get_dict_from_builder(epw_builder)

    qpoints_distance = orm.Float(inputs["qpoints_distance"])
    kpoints_distance_scf = orm.Float(inputs["kpoints_distance_scf"])
    kpoints_factor_nscf = orm.Int(inputs["kpoints_factor_nscf"])
    do_bands_interpolation = orm.Bool(inputs.get("do_bands_interpolation", True))
    kpoints_force_parity = orm.Bool(inputs.get("kpoints_force_parity", False))

    validation_error = validate_inputs(inputs)
    if validation_error is not None:
        raise ValueError(validation_error)

    force_parity = kpoints_force_parity

    reciprocal_points = generate_reciprocal_points(
        structure=structure,
        force_parity=force_parity,
        kpoints_distance_scf=kpoints_distance_scf,
        qpoints_distance=qpoints_distance,
        kpoints_factor_nscf=kpoints_factor_nscf,
    )

    should_run_w90 = should_run_wannier90(w90_parameters=w90_bands.get("wannier90", {}).get("wannier90", {}).get("parameters"))
    with If(should_run_w90.result):
        if "reference_bands" in w90_bands:
            wannier90_run_proxy = Wannier90OptimizeTask(**w90_bands)
        else:
            wannier90_run_proxy = Wannier90BandsTask(**w90_bands)

        w90_t = wannier90_run_proxy._task
        w90_t.inputs["scf"]["kpoints"] = reciprocal_points.kpoints_scf
        w90_t.inputs["nscf"]["kpoints"] = reciprocal_points.kpoints_nscf
        w90_t.inputs["wannier90"]["wannier90"]["kpoints"] = reciprocal_points.kpoints_nscf
        w90_t.inputs["structure"] = structure
        
        updated_parameters = update_wannier90_parameters(
            parameters=w90_bands["wannier90"]["wannier90"]["parameters"],
            kpoints_nscf=reciprocal_points.kpoints_nscf,
        )
        w90_t.inputs["wannier90"]["wannier90"]["parameters"] = updated_parameters.parameters
        
        wannier90_run = wannier90_run_proxy

    phonons_inputs = recursive_merge(ph_base, {
        "qpoints": reciprocal_points.qpoints,
        "ph": {"parent_folder": wannier90_run.scf.remote_folder},
    })
    phonons_run = PhBaseTask(**phonons_inputs)

    kfpoints = create_kpoints_gamma()
    epw_inputs = recursive_merge(epw_base, {
        "structure": structure,
        "parent_folder_ph": phonons_run.remote_folder,
        "parent_folder_nscf": wannier90_run.nscf.remote_folder,
        "kpoints": reciprocal_points.kpoints_nscf,
        "kfpoints": kfpoints.result,
        "qpoints": reciprocal_points.qpoints,
        "qfpoints": kfpoints.result,
    })
    
    if "reference_bands" in w90_bands and w90_bands.get("optimize_disproj"):
        epw_inputs["parent_folder_chk"] = wannier90_run.wannier90_optimal.remote_folder
    else:
        epw_inputs["parent_folder_chk"] = wannier90_run.wannier90.remote_folder
        
    epw_run = EpwBaseTask(**epw_inputs)

    should_run_bands = should_run_epw_bands(
        do_bands_interpolation=do_bands_interpolation,
        epw_parameters=epw_bands.get("parameters"),
    )
    with If(should_run_bands.result):
        epw_bands_inputs = recursive_merge(epw_bands, {
            "structure": structure,
            "parent_folder_epw": epw_run.remote_stash,
            "kpoints": reciprocal_points.kpoints_nscf,
            "qpoints": reciprocal_points.qpoints,
        })
        
        if "bands_kpoints" in w90_bands:
             bands_kpoints_source = w90_bands["bands_kpoints"]
             epw_bands_inputs["qfpoints"] = bands_kpoints_source
             epw_bands_inputs["kfpoints"] = bands_kpoints_source
        else:
             if "reference_bands" in w90_bands and w90_bands.get("optimize_disproj"):
                 band_structure = wannier90_run.wannier90_optimal.band_structure
             else:
                 band_structure = wannier90_run.wannier90.band_structure
             bands_kpoints_task = prepare_epw_bands_kpoints(band_structure)
             epw_bands_inputs["qfpoints"] = bands_kpoints_task.bands_kpoints
             epw_bands_inputs["kfpoints"] = bands_kpoints_task.bands_kpoints
             
        epw_bands_run = EpwBaseTask(**epw_bands_inputs)


    return results(
        retrieved=epw_run.retrieved,
        epw_folder=epw_run.epw_folder,
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
    filepath = files(protocols) / "prep.yaml"

    AdHocProtocol = type("AdHocProtocol", (ProtocolMixin,), {
        "get_protocol_filepath": classmethod(lambda cls: filepath),
        "_validate_override_keys": classmethod(lambda cls, overrides: None)
    })

    return AdHocProtocol.get_protocol_inputs(protocol, overrides)


