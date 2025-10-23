def find_related_calculation(parent_folder_epw):
    """Find the related calculation from a parent folder of an epw calculation."""
    creator = parent_folder_epw.creator
    if creator.process_label == 'EpwCalculation':
        calculation = creator
    elif creator.process_label == 'move_stash':
        calculation = creator.inputs.stash_data.creator
    else:
        raise ValueError(f"Unknown process label: {creator.process_label}")

    if not calculation.process_label == 'EpwCalculation':
        raise ValueError(f"Related calculation is not a valid epw calculation: {calculation.process_label}")

    return calculation