"""WorkGraph implementation of the EPW preparation workflow."""

from __future__ import annotations

from typing import Any

from aiida import orm
from aiida_workgraph import If, WorkGraph, spec, task
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_workgraph.utils import get_dict_from_builder
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90OptimizeWorkChain,
)
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_epw.tools.workchain import get_target_basepath
from aiida_epw.workflows.base import EpwBaseWorkChain

Wannier90BandsTask = task(Wannier90BandsWorkChain)
Wannier90OptimizeTask = task(Wannier90OptimizeWorkChain)
PhBaseTask = task(PhBaseWorkChain)
EpwBaseTask = task(EpwBaseWorkChain)

__all__ = (
    "prep",
    "prep_from_inputs",
    "build_task_inputs",
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


def _as_bool(value: Any) -> bool:
    """Convert Python and AiiDA booleans into a plain boolean."""
    if isinstance(value, orm.Bool):
        return value.value
    return bool(value)


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert AiiDA and plain mapping inputs into a plain dictionary."""
    if value is None:
        return {}
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, dict):
        return value
    return dict(value)


def _create_kpoints_from_distance_node(
    structure: orm.StructureData,
    distance: orm.Float,
    force_parity: orm.Bool,
) -> orm.KpointsData:
    """Create a k-point mesh with the same symmetry handling as the QE helper."""
    from numpy import linalg

    epsilon = 1e-5

    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(
        distance.value,
        force_parity=force_parity.value,
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


def _add_metadata_stash_target_base(inputs: dict[str, Any], code) -> None:
    """Populate a ``metadata.options.stash`` target base when missing."""
    if code is None:
        return

    stash = (
        inputs.setdefault("metadata", {})
        .setdefault("options", {})
        .setdefault("stash", {})
    )
    if "target_base" not in stash:
        stash["target_base"] = get_target_basepath(code.computer)
        stash["stash_mode"] = stash.get("stash_mode", "copy")


def _add_options_stash_target_base(inputs: dict[str, Any], code) -> None:
    """Populate an ``options.stash`` target base when missing."""
    if code is None:
        return

    stash = inputs.setdefault("options", {}).setdefault("stash", {})
    if "target_base" not in stash:
        stash["target_base"] = get_target_basepath(code.computer)
        stash["stash_mode"] = stash.get("stash_mode", "copy")


def _build_wannier90_inputs(
    *,
    codes: dict[str, Any],
    structure: orm.StructureData,
    protocol_inputs: dict[str, Any],
    pseudo_family: Any,
    wannier_projection_type,
    reference_bands: orm.BandsData | None,
    bands_kpoints: orm.KpointsData | None,
) -> dict[str, Any]:
    """Build the static inputs for the Wannier90 task."""
    if reference_bands is not None:
        w90_builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            structure=structure,
            codes=codes,
            pseudo_family=pseudo_family,
            overrides=protocol_inputs.get("w90_bands", {}),
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
            overrides=protocol_inputs.get("w90_bands", {}),
            projection_type=wannier_projection_type,
            bands_kpoints=bands_kpoints,
        )

    w90_bands = get_dict_from_builder(w90_builder)
    if wannier_projection_type == WannierProjectionType.ATOMIC_PROJECTORS_QE:
        w90_bands.pop("projwfc", None)

    w90_bands.pop("structure", None)
    w90_bands.pop("open_grid", None)
    return w90_bands


def _build_ph_inputs(
    *,
    codes: dict[str, Any],
    protocol: str | None,
    protocol_inputs: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the static inputs for the phonon task."""
    ph_base_inputs = protocol_inputs.get("ph_base", {})
    _add_metadata_stash_target_base(ph_base_inputs.setdefault("ph", {}), codes.get("ph"))

    ph_base_builder = PhBaseWorkChain.get_builder_from_protocol(
        codes["ph"], None, protocol, overrides=ph_base_inputs, **kwargs
    )
    ph_base = get_dict_from_builder(ph_base_builder)
    ph_base.pop("clean_workdir", None)
    ph_base.pop("qpoints_distance", None)
    return ph_base


def _build_epw_inputs(
    *,
    codes: dict[str, Any],
    structure: orm.StructureData,
    protocol: str | None,
    protocol_inputs: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the static inputs for the transformation and bands EPW tasks."""
    epw_base: dict[str, Any] = {}
    epw_bands: dict[str, Any] = {}
    epw_code = codes.get("epw")

    for namespace in ("epw_base", "epw_bands"):
        if namespace == "epw_bands" and not protocol_inputs.get(
            "do_bands_interpolation", True
        ):
            continue

        epw_inputs = protocol_inputs.get(namespace, {})
        if namespace == "epw_base":
            _add_options_stash_target_base(epw_inputs, epw_code)

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

    return epw_base, epw_bands


def build_task_inputs(
    *,
    codes: dict,
    structure: orm.StructureData,
    protocol: str | None = None,
    overrides: dict | None = None,
    wannier_projection_type=None,
    reference_bands: orm.BandsData | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Build and validate the static inputs required by the prep WorkGraph."""
    if hasattr(codes, "items"):
        codes = dict(codes.items())
    if overrides and hasattr(overrides, "items"):
        overrides = dict(overrides.items())

    protocol_inputs = get_protocol_inputs(protocol, overrides)
    validation_error = validate_inputs(protocol_inputs)
    if validation_error is not None:
        raise ValueError(validation_error)

    if wannier_projection_type is None:
        wannier_projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE

    pseudo_family = protocol_inputs.pop("pseudo_family", None)
    w90_bands = _build_wannier90_inputs(
        codes=codes,
        structure=structure,
        protocol_inputs=protocol_inputs,
        pseudo_family=pseudo_family,
        wannier_projection_type=wannier_projection_type,
        reference_bands=reference_bands,
        bands_kpoints=bands_kpoints,
    )
    ph_base = _build_ph_inputs(
        codes=codes,
        protocol=protocol,
        protocol_inputs=protocol_inputs,
        kwargs=kwargs,
    )
    epw_base, epw_bands = _build_epw_inputs(
        codes=codes,
        structure=structure,
        protocol=protocol,
        protocol_inputs=protocol_inputs,
        kwargs=kwargs,
    )

    return {
        "w90_bands": w90_bands,
        "ph_base": ph_base,
        "epw_base": epw_base,
        "epw_bands": epw_bands,
        "qpoints_distance": orm.Float(protocol_inputs["qpoints_distance"]),
        "kpoints_distance_scf": orm.Float(protocol_inputs["kpoints_distance_scf"]),
        "kpoints_factor_nscf": orm.Int(protocol_inputs["kpoints_factor_nscf"]),
        "do_bands_interpolation": orm.Bool(
            protocol_inputs.get("do_bands_interpolation", True)
        ),
        "kpoints_force_parity": orm.Bool(
            protocol_inputs.get("kpoints_force_parity", False)
        ),
    }

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
    kpoints_scf = _create_kpoints_from_distance_node(
        structure=structure,
        distance=kpoints_distance_scf,
        force_parity=force_parity,
    )
    qpoints = _create_kpoints_from_distance_node(
        structure=structure,
        distance=qpoints_distance,
        force_parity=force_parity,
    )
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
        bands_plot = _as_dict(w90_parameters).get("bands_plot", False)
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
    do_bands = _as_bool(do_bands_interpolation)
    bands_plot = _as_dict(epw_parameters).get("band_plot", False)

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


@task.calcfunction(outputs=spec.namespace(retrieved=Any, epw_folder=Any))
def results(retrieved, epw_folder):
    """Expose the final outputs following the original ``results`` step."""
    return {
        "retrieved": retrieved,
        "epw_folder": epw_folder,
    }


def prep_from_inputs(
    *,
    structure: orm.StructureData,
    inputs: dict[str, Any],
    use_wannier90_optimize: bool,
):
    """Create the prep WorkGraph from pre-built static task inputs."""
    w90_bands = inputs["w90_bands"]
    ph_base = inputs["ph_base"]
    epw_base = inputs["epw_base"]
    epw_bands = inputs["epw_bands"]

    with WorkGraph(
        name="prep",
        outputs=spec.namespace(retrieved=Any, epw_folder=Any),
    ) as wg:
        reciprocal_points = generate_reciprocal_points(
            structure=structure,
            force_parity=inputs["kpoints_force_parity"],
            kpoints_distance_scf=inputs["kpoints_distance_scf"],
            qpoints_distance=inputs["qpoints_distance"],
            kpoints_factor_nscf=inputs["kpoints_factor_nscf"],
        )

        should_run_w90 = should_run_wannier90(
            w90_parameters=w90_bands.get("wannier90", {}).get("wannier90", {}).get("parameters")
        )
        with If(should_run_w90.result):
            if use_wannier90_optimize:
                wannier90_run_proxy = Wannier90OptimizeTask(**w90_bands)
                w90_class_name = "Wannier90OptimizeWorkChain"
            else:
                wannier90_run_proxy = Wannier90BandsTask(**w90_bands)
                w90_class_name = "Wannier90BandsWorkChain"

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

        if (
            w90_class_name == "Wannier90OptimizeWorkChain"
            and _as_bool(w90_bands.get("optimize_disproj", False))
        ):
            epw_inputs["parent_folder_chk"] = wannier90_run.wannier90_optimal.remote_folder
        else:
            epw_inputs["parent_folder_chk"] = wannier90_run.wannier90.remote_folder

        epw_run = EpwBaseTask(**epw_inputs)

        should_run_bands = should_run_epw_bands(
            do_bands_interpolation=inputs["do_bands_interpolation"],
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
                bands_kpoints = extract_kpoints_path(
                    band_structure=wannier90_run.band_structure
                ).result
                epw_bands_inputs["qfpoints"] = bands_kpoints
                epw_bands_inputs["kfpoints"] = bands_kpoints

            EpwBaseTask(**epw_bands_inputs)

        final_results = results(
            retrieved=epw_run.retrieved,
            epw_folder=epw_run.remote_folder,
        )
        wg.outputs.retrieved = final_results.retrieved
        wg.outputs.epw_folder = final_results.epw_folder

    return wg


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
    """Build the EPW preparation WorkGraph."""
    inputs = build_task_inputs(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        wannier_projection_type=wannier_projection_type,
        reference_bands=reference_bands,
        bands_kpoints=bands_kpoints,
        **kwargs,
    )
    return prep_from_inputs(
        structure=structure,
        inputs=inputs,
        use_wannier90_optimize=reference_bands is not None,
    )
