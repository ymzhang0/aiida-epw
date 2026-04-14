"""Helpers for tracing workflow parent folders."""


def _normalize_structure_component(value):
    """Normalize nested structure data for tolerant equality checks."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_normalize_structure_component(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_structure_component(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _normalize_structure_component(item)
            for key, item in sorted(value.items())
        }
    return value


def structures_match(left, right) -> bool:
    """Return whether two ``StructureData`` nodes describe the same structure."""
    left_signature = {
        "cell": _normalize_structure_component(left.cell),
        "pbc": tuple(bool(value) for value in left.pbc),
        "kinds": _normalize_structure_component(left.base.attributes.get("kinds", [])),
        "sites": _normalize_structure_component(left.base.attributes.get("sites", [])),
    }
    right_signature = {
        "cell": _normalize_structure_component(right.cell),
        "pbc": tuple(bool(value) for value in right.pbc),
        "kinds": _normalize_structure_component(right.base.attributes.get("kinds", [])),
        "sites": _normalize_structure_component(right.base.attributes.get("sites", [])),
    }
    return left_signature == right_signature


def get_parent_folder_calculation(parent_folder):
    """Return the calculation that produced a remote or stashed parent folder."""
    creator = parent_folder.creator
    if creator is None:
        raise ValueError("The provided parent folder does not have a creator.")

    if creator.process_label == "move_stash":
        creator = creator.inputs.stash_data.creator

    return creator


def get_parent_ph_calculation(parent_folder_ph):
    """Return the ``PhCalculation`` that produced a phonon parent folder."""
    calculation = get_parent_folder_calculation(parent_folder_ph)

    if calculation.process_label != "PhCalculation":
        raise ValueError(
            "`parent_folder_ph` must be created by a `PhCalculation` or its stashed "
            f"remote folder, got `{calculation.process_label}`."
        )

    return calculation


def get_parent_ph_qpoints(parent_folder_ph):
    """Return the q-point mesh associated with a phonon parent folder."""
    calculation = get_parent_ph_calculation(parent_folder_ph)
    qpoints = getattr(calculation.inputs, "qpoints", None)

    if qpoints is None:
        raise ValueError(
            "The provided `parent_folder_ph` does not expose input `qpoints`."
        )

    return qpoints


def validate_parent_ph_inputs(parent_folder_ph, structure):
    """Validate a phonon parent folder against the target EPW structure."""
    qpoints = get_parent_ph_qpoints(parent_folder_ph)
    ph_calculation = get_parent_ph_calculation(parent_folder_ph)

    parent_pw_folder = getattr(ph_calculation.inputs, "parent_folder", None)
    if parent_pw_folder is None:
        raise ValueError(
            "The provided `parent_folder_ph` does not expose the parent PW folder "
            "needed to validate its structure."
        )

    parent_pw_calculation = get_parent_folder_calculation(parent_pw_folder)
    parent_structure = getattr(parent_pw_calculation.inputs, "structure", None)
    if parent_structure is None:
        raise ValueError(
            "The PW calculation linked to `parent_folder_ph` does not expose input "
            "`structure`."
        )

    if not structures_match(parent_structure, structure):
        raise ValueError(
            "The structure used to generate `parent_folder_ph` does not match the "
            "current `EpwPrepWorkChain.structure`."
        )

    return qpoints


def find_related_calculation(parent_folder_epw):
    """Find the related calculation from a parent folder of an epw calculation."""
    calculation = get_parent_folder_calculation(parent_folder_epw)

    if not calculation.process_label == "EpwCalculation":
        raise ValueError(
            f"Related calculation is not a valid epw calculation: {calculation.process_label}"
        )

    return calculation


def format_subprocess_failure(node, process_label=None):
    """Return a readable failure message for a subprocess node."""
    label = process_label or getattr(node, "process_label", node.__class__.__name__)
    message = f"{label}<{node.pk}> failed with exit status {node.exit_status}"
    exit_message = getattr(node, "exit_message", None)

    if exit_message:
        message = f"{message}: {exit_message}"

    return message

def get_target_basepath(computer):
    """Set the target basepath for the stash folder."""
    from pathlib import Path
    if computer.transport_type == "core.local":
        target_basepath = Path(computer.get_workdir(), "stash").as_posix()
    elif computer.transport_type.startswith("core.ssh"):
        workdir = computer.get_workdir()
        if "{username}" in workdir:
            username = computer.get_configuration().get("username")
            if not username:
                try:
                    from aiida.orm import User
                    auth_info = computer.get_authinfo(User.objects.get_default())
                    username = auth_info.get_auth_params().get("username")
                except Exception:
                    pass
            if not username:
                raise ValueError(f"Could not determine username to format workdir for computer '{computer.label}'")
            target_basepath = Path(workdir.format(username=username), "stash").as_posix()
        else:
            target_basepath = Path(workdir, "stash").as_posix()
    else:
        raise ValueError(f"Unsupported transport type: {computer.transport_type}")
    return target_basepath
