************************
TransportWorkChain
************************

This tutorial will guide you through running a complete `EpwTransportWorkChain` to calculate the transport properties of a material.


Step 1: Prepare the input EpwPrepWorkChain
====================================

You need to have a successfully run EpwPrepWorkChain as the input of the TransportWorkChain.

You may refer to this tutorial: :ref:`supercon` to learn how to do the preparation work.


Step 2: Create the builder for TransportWorkChain
======================================================

We will use the `get_builder_from_protocol` factory method to easily set up the inputs. We will run a "fast" calculation from scratch.

.. code-block:: python

   from aiida_epw.workflows import TransportWorkChain
   from aiida_wannier90_workflows.common.types import WannierProjectionType
   import yaml

   builder = EpwTransportWorkChain.get_builder_from_protocol(
       epw_code=epw_code,
       parent_epw=epw_workchain,
       protocol='fast',  # Use the 'fast' protocol for a quick test
       # We can provide overrides for specific parameters if needed
       overrides={
            'epw_interp': {
                'metadata': {'options': {'max_wallclock_seconds': 1800}}
            },
            'transport': {
                'metadata': {'options': {'max_wallclock_seconds': 1800}}
            },
        },
   )

   # You can modify the builder further if needed, e.g., for cleanup
   builder.clean_workdir = orm.Bool(True)

Please refer to the override.yaml inside the protocols folder for the structure of the overrides.

Step 3: Submit and run the calculation
=======================================

Use the AiiDA engine to run the workflow and get the results.

.. code-block:: python

   result, workchain_node = engine.run_get_node(builder)

Step 4: Inspect the results
===========================

to be added...