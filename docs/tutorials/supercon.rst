.. _supercon:

==================
SuperConWorkChain
==================


This tutorial will guide you through running a complete `EpwSuperconWorkChain` to calculate the superconducting properties of a material.


Step 1: Prepare the input EpwPrepWorkChain
=======================================

You need to have a successfully run EpwPrepWorkChain as the input of the SuperConWorkChain.

You may refer to this tutorial: :ref:`prep` to learn how to run an EpwPrepWorkChain.

Step 2: Create the builder for SuperConWorkChain
=================================================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will use the 'fast' protocol for a quick test.

.. code-block:: python

   from aiida_epw.workflows import EpwSuperconWorkChain

   builder = SuperConWorkChain.get_builder_from_protocol(
       epw_code=epw_code,
       structure=structure,
       protocol='fast',  # Use the 'fast' protocol for a quick test
       # We can provide overrides for specific parameters if needed
       parent_epw=epw_workchain,
       overrides={
            'epw_interp': {
                'metadata': {'options': {'max_wallclock_seconds': 1800}}
            },
            'epw_final_iso': {
                'metadata': {'options': {'max_wallclock_seconds': 1800}}
            },
            'epw_final_aniso': {
                'metadata': {'options': {'max_wallclock_seconds': 1800}}
            },
            'interpolation_distance': [0.2, 0.1, 0.08, 0.06, 0.05],
            'convergence_threshold': 0.1,
            'always_run_final': True,
        }
    )

In this example, we provide with the builder "epw_workchain" which is the previous EpwPrepWorkChain node.

Then we set up the interpolation distance and the convergence threshold.

SuperConWorkChain will iterate over the interpolation distances and calculate the spertral function at each interpolation distance.

Then it will calculate the Allen-Dynes Tc from the spectral function using the Allen-Dynes formula.

It will check the convergence of the Allen-Dynes Tc (in our case, it is 0.1). 

We set the "always_run_final" to True, so that the final isotropic and anisotropic calculations will always be run even if the convergence is achieved.

Step 3: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   node, results = engine.run_get_node(builder)

Step 4: Inspect the results
============================

Once the `SuperConWorkChain` has finished successfully, you can inspect its outputs.

.. code-block:: python

    print(f"WorkChain finished with status: {node.process_state}")
    print(f"Available outputs: {results.keys()}")

    # Get the final Allen-Dynes Tc from the 'epw_final_iso' and 'epw_final_aniso' sub-process results

    Allen_Dynes_Tc = results['Allen_Dynes_Tc']
    print(f"Calculated Allen-Dynes Tc = {Allen_Dynes_Tc:.2f} K")

    # You can also get the isotropic Tc from the 'iso' sub-process results
    iso_tc = results['iso_tc']
    print(f"Calculated Tc from isotropic Migdal-Eliashberg equation = {iso_tc:.2f} K")

    # You can also get the anisotropic Tc from the 'epw_final_aniso' sub-process results
    aniso_tc = results['aniso_tc']
    print(f"Calculated Tc from anisotropic Migdal-Eliashberg equation = {aniso_tc:.2f} K")

The anisotropic Tc will not automatically be outputted. You need to plot the gap functions for an extropolation of it..

This concludes the quick start tutorial. For more advanced topics, such as restarting calculations or using the submission controller, please refer to the User Guide.