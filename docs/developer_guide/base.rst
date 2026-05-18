.. _base:

=======================
EpwBaseWorkChain
=======================

This is the BaseRestartWorkChain equipped with ProtocolMixin [from aiida-quantumespresso] for a EpwCalculation, sharing similar structure as PwBaseWorkChain, PhBaseWorkChain.

It will,

- Provide a builder generated from a protocol for the submission of the EpwCalculation process.
- Check the parent folder of finished wannier90 calculation and a phonon calculation; check the compatibility of the coarse k/q grids.
- Automatically generate the fine k/q grids for epw interpolation.
- Automatically restart the calculation upon failures.
- TODO: Handle the EPW errors automatically.

------------------------------
In `define()` method:
------------------------------

Firstly, it expose the whole inputs namespace of the EpwCalculation to the top-level namespace of the work chain.

This behaves differently from PwBaseWorkChain.

It is an experimental feature mainly to avoid the duplicated definition of the inputs ports.

The price to pay is that we should provide an additional `options` input port since the `metadata` input port of the EpwCalculation is overridden by the homonymous port in EpwBaseWorkChain.


-------------------------------------------
In `get_builder_from_protocol()` method:
-------------------------------------------

In the function we subsequently construct the inputs of the subprocesses:

- Wannier90BandsWorkChain/Wannier90OptimizeWorkChain
- PhBaseWorkChain
- EpwBaseWorkChain [for transformation from coarse-grid Bloch basis to Wannier basis]
- EpwBaseWorkChain [optional, for interpolation of the electron and phonon band structures]

We decide whether to do a Wannier90BandsWorkChain or a Wannier90OptimizeWorkChain by the presence of the reference_bands input port.

If we have a reference_bands, we will do a optimization over the projection window (dis_proj_min/max)

Otherwise, it's simply a single wannierization in Wannier90BandsWorkChain.

It should be noted that if we choose to use atomic orbitals as projectors, we should exclude projwfc namespace to avoid possible inputs validation.

.. note::
    If the quality of wannierization is not good, it is usually due to the insufficient projectors.
    This can be solved by providing external projectors, which is implemented in the current `aiida-wannier90-workflows` package.
    And we should then adapt this in our aiida-epw package.

-------------------------------------------
In the workflow logic:
-------------------------------------------

In this work chain, we mainly take into account two types of works:

- The calculation from scratch provided with the parent folders of successful Wannier90Calculation and PhononCalculation work chains.
- The calculation from restart provided with the parent folder of a previous EpwCalculation.

For the first type, it will validate the parent folders.

- If the phonon parent folders are valid, it will use the qpoints from the phonon parent folders.

- If the wannier90 parent folders are valid, it will use the kpoints from the wannier90 parent folders.

- If they are not compatible, it will re-generate the kpoints of the wannier90 work chain based on the qpoints of the phonon work chain. Then it will re-run the wannier90 work chain.

- If the wannier90 parent folders are not valid, it will run the wannier90 work chain from scratch.

If none of the parent folders are provided or are not valid, it will run the Wannier90 and Phonon work chains from scratch.
