def get_parent_folder_calculation(parent_folder):
    """Return the calculation that produced a remote or stashed parent folder."""
    creator = parent_folder.creator
    if creator is None:
        raise ValueError("The provided parent folder does not have a creator.")

    if creator.process_label == "move_stash":
        creator = creator.inputs.stash_data.creator

    return creator


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