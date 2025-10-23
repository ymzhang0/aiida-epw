.. _prep:

=================
EpwPrepWorkChain
=================

This tutorial will guide you through running a complete `EpwPrepWorkChain` to calculate electron-phonon coupling matrix on Wannier representation.

Step 1: Setup your AiiDA environment
=======================================

First, make sure you are in a running `verdi shell` or have loaded the AiiDA profile in your Python script.

.. code-block:: python

   from aiida import orm, engine

   # Load all the necessary codes
   codes = {
       'pw': orm.load_code('pw-7.5@my_cluster'),
       'ph': orm.load_code('ph-7.5@my_cluster'),
       'epw': orm.load_code('epw-6.0@my_cluster'),
       'wannier90': orm.load_code('wannier90-p2w_ham@my_cluster'),
       'pw2wannier90': orm.load_code('pw2wannier90-7.5@my_cluster'),
   }

.. note::

    It is recommended to use PDWF to automate the wannierization. And one can find the installation here: https://github.com/qiaojunfeng/wannier90/tree/p2w_ham

Step 2: Prepare the input structure
====================================

Load the crystal structure you want to calculate.

For electron-phonon coupling calculation, Pb and MgB$_2$ are good examples. You can find the structures in the `examples/structures/` folder within the package.

.. code-block:: python

   # Load a structure from files
   structure =  read_structure_from_file(
        package / 'examples/structures/Pb.xsf'
        )
    or
    structure =  read_structure_from_file(
        package / 'examples/structures/MgB2.xsf'
        )

Step 3: Create the builder for EpwPrepWorkChain
=============================================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will run a "fast" calculation from scratch.

.. code-block:: python

   from aiida_epw.workflows import EpwPrepWorkChain
   from aiida_wannier90_workflows.common.types import WannierProjectionType

   builder = EpwPrepWorkChain.get_builder_from_protocol(
       codes=codes,
       structure=structure,
       protocol='fast',  # Use the 'fast' protocol for a quick test
       # We can provide overrides for specific parameters if needed
       overrides={
            'w90_bands': {
                'scf': {
                    'pw': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
                    },
                'nscf': {
                    'pw': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
                    },
                'wannier90': {
                    'wannier90': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
                    },
                'pw2wannier90': {
                    'pw2wannier90': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
                },
            },
            'ph_base': {
                'ph': {'metadata': {'options': {'max_wallclock_seconds': 1800}}}
            },
            'epw_base':{
                'metadata': {'options': {'max_wallclock_seconds': 1800}},
            },
            'epw_bands':{
                'metadata': {'options': {'max_wallclock_seconds': 1800}},
            },
        },
       # Specify the wannierization scheme, here it is PDWF.
       wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
       # Specify the script to convert the wannier90 checkpoint file to the ukk format that is used for EPW.
       w90_chk_to_ukk_script = w90_script,
   )

   # You can modify the builder further if needed, e.g., for cleanup
   builder.clean_workdir = orm.Bool(True)


Step 4: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   node, results = engine.run_get_node(builder)

Step 5: Inspect the results
============================

The EpwWorkChain will not output very meaningful direct results. It is more like a preparation step for the following SuperConWorkChain/TransportWorkChain. 

It will output the following intermediate results:

- `retrieved`: The retrieved folder containing the results of the calculation.
- `epw_folder`: The remote folder containing the results of the calculation.

You can inspect the results by accessing the `retrieved` and `epw_folder` outputs.

.. code-block:: python

   print(results['retrieved'])
   print(results['epw_folder'])