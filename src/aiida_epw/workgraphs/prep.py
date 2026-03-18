"""WorkGraph implementation of the EPW preparation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_workgraph import If, spec, task
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90OptimizeWorkChain,
)
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_epw.tools.workchain import get_parent_folder_calculation, get_target_basepath
from aiida_epw.workflows.base import EpwBaseWorkChain
from aiida_epw.workflows.prep import should_run_bands_interpolation, validate_inputs

Wannier90BandsTask = task(Wannier90BandsWorkChain)
Wannier90OptimizeTask = task(Wannier90OptimizeWorkChain)
PhBaseTask = task(PhBaseWorkChain)
EpwBaseTask = task(EpwBaseWorkChain)

__all__ = (
    "prep",
)

@task.calcfunction(
    outputs=spec.namespace(
        kpoints_scf=Any,
        qpoints=Any,
        kpoints_nscf=Any,
    )
)
def generate_reciprocal_points(structure, force_parity, kpoints_distance_scf, qpoints_distance, kpoints_factor_nscf):
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
def should_run_wannier90(w90_bands) -> bool:
    """Mirror the outline guard for the Wannier90 branch."""
    return w90_bands is not None


@task.calcfunction(outputs=spec.namespace(parameters=Any))
def update_wannier90_parameters(parameters, kpoints_nscf):
    """Update the Wannier90 ``mp_grid`` to match the NSCF mesh."""
    updated = parameters.get_dict()
    updated["mp_grid"] = kpoints_nscf.get_kpoints_mesh()[0]
    return {"parameters": orm.Dict(updated)}


@task.graph(
    outputs=spec.namespace(
        scf_remote=Any,
        nscf_remote=Any,
        chk_folder=Any,
        band_structure=Any,
    )
)
def run_wannier90(
    structure,
    w90_bands,
    kpoints_scf,
    kpoints_nscf,
):
    """Run the Wannier workflow step of the EPW preparation graph."""
    updated_parameters = update_wannier90_parameters(
        w90_bands["wannier90"]["wannier90"]["parameters"],
        kpoints_nscf,
    ).parameters

    w90_inputs = recursive_merge(
        w90_bands,
        {
            "structure": structure,
            "scf": {"kpoints": kpoints_scf},
            "nscf": {"kpoints": kpoints_nscf},
            "wannier90": {
                "wannier90": {
                    "kpoints": kpoints_nscf,
                    "parameters": updated_parameters,
                }
            },
        },
    )

    if "reference_bands" in w90_bands:
        w90 = Wannier90OptimizeTask(**w90_inputs)
    else:
        w90 = Wannier90BandsTask(**w90_inputs)

    if "reference_bands" in w90_bands and w90_bands.get("optimize_disproj"):
        chk_folder = w90.wannier90_optimal.remote_folder
    else:
        chk_folder = w90.wannier90.remote_folder

    inspected = inspect_wannier90(
        scf_remote=w90.scf.remote_folder,
        nscf_remote=w90.nscf.remote_folder,
        chk_folder=chk_folder,
        band_structure=w90.band_structure,
    )
    return {
        "scf_remote": inspected.scf_remote,
        "nscf_remote": inspected.nscf_remote,
        "chk_folder": inspected.chk_folder,
        "band_structure": inspected.band_structure,
    }


def find_parent_workchain(node, label):
    """Find the parent workchain by traversing up `caller` links."""
    from aiida.common.links import LinkType
    # Get creating node
    links = node.get_incoming(link_type=LinkType.CREATE).all()
    if not links:
        return None
    current = links[0].node
    while current is not None:
        if getattr(current, "process_label", "") == label:
            return current
        current = current.caller
    return None


@task.calcfunction(
    outputs=spec.namespace(
        scf_remote=Any,
        nscf_remote=Any,
        chk_folder=Any,
        band_structure=Any,
    )
)
def inspect_wannier90(scf_remote, nscf_remote, chk_folder, band_structure):
    """Verify that the wannier90 workflow finished successfully."""
    from aiida.engine import ExitCode
    
    workchain = find_parent_workchain(scf_remote, "Wannier90BandsWorkChain") or \
                find_parent_workchain(scf_remote, "Wannier90OptimizeWorkChain")
                
    if workchain and not workchain.is_finished_ok:
         return ExitCode(404, message=f"The `Wannier90BandsWorkChain` sub process failed with exit_status {workchain.exit_status}")

    return {
        "scf_remote": scf_remote,
        "nscf_remote": nscf_remote,
        "chk_folder": chk_folder,
        "band_structure": band_structure,
    }


@task.calcfunction()
def extract_parent_ph_qpoints(parent_folder_ph):
    """Extract the q-point mesh used by the parent ``PhCalculation``."""
    parent_calculation = get_parent_folder_calculation(parent_folder_ph)
    if parent_calculation.process_label != "PhCalculation":
        raise ValueError("The provided parent folder does not come from `PhCalculation`.")

    return parent_calculation.inputs.qpoints


@task.graph(outputs=spec.namespace(remote_folder=Any))
def run_ph(ph_base, qpoints, scf_remote):
    """Run the phonon workflow step of the EPW preparation graph."""
    ph_inputs = recursive_merge(ph_base, {"qpoints": qpoints})

    ph_inputs = recursive_merge(
        ph_inputs,
        {"ph": {"parent_folder": scf_remote}},
    )

    ph = PhBaseTask(**ph_inputs)
    inspected = inspect_ph(remote_folder=ph.remote_folder)
    return {"remote_folder": inspected.remote_folder}


@task.calcfunction(outputs=spec.namespace(remote_folder=Any))
def inspect_ph(remote_folder):
    """Verify that the `PhBaseWorkChain` finished successfully."""
    from aiida.engine import ExitCode
    
    workchain = find_parent_workchain(remote_folder, "PhBaseWorkChain")
    if workchain and not workchain.is_finished_ok:
         return ExitCode(403, message=f"The `PhBaseWorkChain` sub process failed with exit_status {workchain.exit_status}")

    return {"remote_folder": remote_folder}


@task.calcfunction()
def create_kpoints_gamma():
    """Create the gamma-only mesh used by the transformation EPW run."""
    gamma = orm.KpointsData()
    gamma.set_kpoints_mesh([1, 1, 1])
    return gamma


@task.graph(
    outputs=spec.namespace(
        retrieved=Any,
        epw_folder=Any,
        epw_parent=Any,
    )
)
def run_epw(
    structure,
    epw_base,
    qpoints,
    kpoints_nscf,
    ph_remote,
    nscf_remote,
    chk_folder,
):
    """Run the transformation EPW step of the EPW preparation graph."""
    kfpoints = create_kpoints_gamma().result

    epw_inputs = recursive_merge(
        epw_base,
        {
            "structure": structure,
            "parent_folder_ph": ph_remote,
            "parent_folder_nscf": nscf_remote,
            "parent_folder_chk": chk_folder,
            "kpoints": kpoints_nscf,
            "kfpoints": kfpoints,
            "qpoints": qpoints,
            "qfpoints": kfpoints,
        },
    )

    epw = EpwBaseTask(**epw_inputs)
    epw = EpwBaseTask(**epw_inputs)
    inspected = inspect_epw(
        retrieved=epw.retrieved,
        epw_folder=epw.remote_stash,
        epw_parent=epw.remote_stash,
    )
    return {
        "retrieved": inspected.retrieved,
        "epw_folder": inspected.epw_folder,
        "epw_parent": inspected.epw_parent,
    }


@task.calcfunction(
    outputs=spec.namespace(
        retrieved=Any,
        epw_folder=Any,
        epw_parent=Any,
    )
)
def inspect_epw(retrieved, epw_folder, epw_parent):
    """Verify that the `EpwBaseWorkChain` finished successfully."""
    from aiida.engine import ExitCode
    
    workchain = find_parent_workchain(epw_folder, "EpwBaseWorkChain")
    if workchain and not workchain.is_finished_ok:
         return ExitCode(405, message=f"The `EpwBaseWorkChain` sub process failed with exit_status {workchain.exit_status}")

    return {
        "retrieved": retrieved,
        "epw_folder": epw_folder,
        "epw_parent": epw_parent,
    }


@task()
def should_run_epw_bands(do_bands_interpolation=None, epw_bands=None) -> bool:
    """Mirror the outline guard for the EPW bands interpolation branch."""
    inputs = {}
    if do_bands_interpolation is not None:
        inputs["do_bands_interpolation"] = do_bands_interpolation
    if epw_bands is not None:
        inputs["epw_bands"] = epw_bands
    return should_run_bands_interpolation(inputs)


@task.calcfunction()
def extract_kpoints_path(band_structure):
    """Convert a ``BandsData`` output into a standalone ``KpointsData`` node."""
    kpoints = orm.KpointsData()
    kpoints.set_cell(band_structure.cell, band_structure.pbc)

    points, weights = band_structure.get_kpoints(also_weights=True)
    labels = band_structure.labels if band_structure.labels else None
    kpoints.set_kpoints(points, weights=weights, labels=labels)

    return kpoints


@task.graph(outputs=spec.namespace(bands_kpoints=Any))
def prepare_epw_bands_kpoints(w90_bands, band_structure):
    """Prepare the bands interpolation k-points using the original workflow priority."""
    if "bands_kpoints" in w90_bands:
        return {"bands_kpoints": w90_bands["bands_kpoints"]}

    return {"bands_kpoints": extract_kpoints_path(band_structure).result}


@task.graph(outputs=spec.namespace(retrieved=Any, epw_folder=Any))
def run_epw_bands(
    structure,
    w90_bands,
    epw_bands,
    kpoints_nscf,
    qpoints,
    band_structure,
    epw_parent,
):
    """Run the EPW bands interpolation step of the EPW preparation graph."""
    bands_kpoints = prepare_epw_bands_kpoints(
        w90_bands=w90_bands,
        band_structure=band_structure,
    ).bands_kpoints

    epw_bands_inputs = recursive_merge(
        epw_bands,
        {
            "structure": structure,
            "parent_folder_epw": epw_parent,
            "kpoints": kpoints_nscf,
            "qpoints": qpoints,
            "qfpoints": bands_kpoints,
            "kfpoints": bands_kpoints,
        },
    )

    epw_bands_task = EpwBaseTask(**epw_bands_inputs)
    epw_bands_task = EpwBaseTask(**epw_bands_inputs)
    inspected = inspect_epw_bands(
        retrieved=epw_bands_task.retrieved,
        epw_folder=epw_bands_task.remote_stash,
    )
    return {
        "retrieved": inspected.retrieved,
        "epw_folder": inspected.epw_folder,
    }


@task.calcfunction(outputs=spec.namespace(retrieved=Any, epw_folder=Any))
def inspect_epw_bands(retrieved, epw_folder):
    """Verify that the `EpwBandsWorkChain` finished successfully."""
    from aiida.engine import ExitCode
    
    workchain = find_parent_workchain(epw_folder, "EpwBandsWorkChain")
    if workchain and not workchain.is_finished_ok:
         return ExitCode(406, message=f"The `EpwBandsWorkChain` sub process failed with exit_status {workchain.exit_status}")

    return {
        "retrieved": retrieved,
        "epw_folder": epw_folder,
    }


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

    w90_bands = w90_builder._inputs(prune=False)
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
    ph_base = ph_base_builder._inputs(prune=False)
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
            epw_base = epw_builder._inputs(prune=False)
        else:
            epw_bands = epw_builder._inputs(prune=False)

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
        kpoints_distance_scf=kpoints_distance_scf,
        qpoints_distance=qpoints_distance,
        kpoints_factor_nscf=kpoints_factor_nscf,
    )

    should_run_w90 = should_run_wannier90(w90_bands=w90_bands)
    with If(should_run_w90.result):
        wannier90_run = run_wannier90(
            structure=structure,
            w90_bands=w90_bands,
            kpoints_scf=reciprocal_points.kpoints_scf,
            kpoints_nscf_mesh=reciprocal_points.kpoints_nscf_mesh,
            kpoints_nscf_explicit=reciprocal_points.kpoints_nscf_explicit,
        )
        pass

    phonons_run = run_ph(
        ph_base=ph_base,
        qpoints=reciprocal_points.qpoints,
        scf_remote=wannier90_run.scf_remote,
    )

    epw_run = run_epw(
        structure=structure,
        epw_base=epw_base,
        qpoints=reciprocal_points.qpoints,
        kpoints_nscf_mesh=reciprocal_points.kpoints_nscf_mesh,
        ph_remote=phonons_run.remote_folder,
        nscf_remote=wannier90_run.nscf_remote,
        chk_folder=wannier90_run.chk_folder,
    )

    should_run_bands = should_run_epw_bands(
        do_bands_interpolation=do_bands_interpolation,
        epw_bands=epw_bands,
    )
    with If(should_run_bands.result):
        epw_bands_run = run_epw_bands(
            structure=structure,
            w90_bands=w90_bands,
            epw_bands=epw_bands,
            kpoints_nscf_mesh=reciprocal_points.kpoints_nscf_mesh,
            qpoints=reciprocal_points.qpoints,
            band_structure=wannier90_run.band_structure,
            epw_parent=epw_run.epw_parent,
        )

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


