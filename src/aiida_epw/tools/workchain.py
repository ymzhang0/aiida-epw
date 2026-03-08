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
