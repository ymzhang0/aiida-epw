=======================
SuperConWorkChain
=======================

This work chain is used to compute the superconductor critical temperature on different level of theories.

It is an aggregation of the following work chains:

- EpwBaseWorkChain: iteration over different fine k/q grids to converge the Allen-Dynes Tc.
- [EpwBaseWorkChain]: Optional, calculation of isotropic Migdal-Eliashberg equation
- [EpwBaseWorkChain]: Optional, calculation of anisotropic Migdal-Eliashberg equation

It follows the following steps:

.. code-block:: python
    
    spec.outline(
        cls.setup,
        while_(cls.should_run_conv)(
            cls.run_interp_epw,
            cls.inspect_interp_epw,
        ),
        if_(cls.should_run_final)(
            cls.run_final_epw_iso,
            cls.inspect_final_epw_iso,
            cls.run_final_epw_aniso,
            cls.inspect_final_epw_aniso,
        ),

        cls.results
    )

