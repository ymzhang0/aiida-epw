#!/usr/bin/env python
"""Run a ``EpwA2fWorkChain`` for spectral function on lead.

Usage: ./01_a2f.py
"""
import click
from typing import Dict
from aiida import cmdline, orm

from aiida_epw_workflows.workflows.a2f import EpwA2fWorkChain

from aiida_epw_workflows.cli.params import RUN
from aiida_epw_workflows.utils.structure import read_structure
from aiida_epw_workflows.utils.workflows.builder.serializer import print_builder
from aiida_epw_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_epw_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)


def submit(
    codes: Dict[str, orm.Code],
    structure: orm.StructureData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``EpwA2fWorkChain`` to calculate spectral function on lead."""
    builder = EpwA2fWorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol='fast',
    )

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 8,
        "npool": 4,
    }
    set_parallelization(builder, parallelization, process_class=EpwA2fWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODES(help="The pw.x code identified by its ID, UUID or label.")
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@click.argument("filename", type=click.Path(exists=True))
@RUN()
def cli(filename, codes, group, run):
    """Run a ``EpwA2fWorkChain`` to calculate spectral function on lead.

    FILENAME: a crystal structure file, e.g., ``input_files/GaAs.xsf``.
    """
    struct = read_structure(filename, store=True)
    # struct = orm.load_node(126831)
    submit(codes, struct, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
