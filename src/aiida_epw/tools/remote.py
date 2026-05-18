from aiida.engine import calcfunction
from aiida.orm import RemoteStashFolderData, Str, Computer
from pathlib import Path

@calcfunction
def move_stash(
    stash_data: RemoteStashFolderData,
    target_computer_label: Str,
    target_remote_path: Str,
) -> RemoteStashFolderData:
    """Create a new RemoteStashFolderData node pointing to an already existing folder
    on another computer.

    Warning:
        This function does NOT copy files. It only creates the AiiDA data node.
    """

    source_path = Path(stash_data.target_basepath)
    source_list = stash_data.source_list
    stash_mode = stash_data.stash_mode
    suffix = Path(*source_path.parts[-3:])

    target_path = Path(target_remote_path.value) / suffix

    computer = Computer.collection.get(label=target_computer_label.value)

    target_stash = RemoteStashFolderData(
        computer=computer,
        stash_mode=stash_mode,
        target_basepath=str(target_path),
        source_list=source_list,
    )

    return target_stash