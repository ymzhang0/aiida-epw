.. _overview:

===================
Project Structure
===================

This page provides an overview of the ``aiida-epw`` repository structure, explaining the role of each key directory and module. The layout is based on standard AiiDA plugin conventions to facilitate maintainability and collaboration.

High-Level Overview
-------------------

The project follows a layered architecture to maximize code reuse and separate concerns.

.. code-block:: text

   aiida-epw/
   ├── aiida_epw/
   │   ├── calculations/
   │   │   └── epw.py
   │   ├── workflows/
   │   │   ├── base.py
   │   │   ├── epw.py
   │   │   └── supercon.py
   │   ├── parsers/
   │   │   └── epw.py
   ├── controllers/
   │   └── supercon.py
   │   └── transport.py
   ├── docs/
   ├── tests/
   └── pyproject.toml

Module Breakdown
=================

``aiida_epw/workflows/``
**********************************

This is the core of the plugin, containing all the AiiDA ``WorkChain`` definitions.

``base.py``: `EpwBaseWorkChain`
================================
This is the lowest-level workchain wrapper around a single ``EpwCalculation``. It should include robust error handling and restart logic for a single ``epw.x`` run, but does not handle complex, multi-step physics workflows.

``epw.py``: `EpwB2WWorkChain`
===============================
This is a wrapper around the `Wannier90OptimizeWorkChain`, `PhBaseWorkChain` and `EpwBaseWorkChain`. It internally runs the full `Wannier90OptimizeWorkChain`, a `PhBaseWorkChain`, and a final `EpwBaseWorkChain` to generate the electron-phonon matrix elements in the Wannier representation (``.epmatwp`` files). It can be run standalone or as a preparatory step for other calculations.

``supercon.py``: `EpwSuperconWorkChain`
=========================================
This is the highest-level **orchestrator** workchain. It coordinates a complex computational pipeline by **composing** the other workchains:

1.  It runs the `EpwB2WWorkChain` **once** to get the shared Wannier-representation matrix elements.
2.  It then uses the output of this single `b2w` run to launch the `EpwBandWorkChain` to get the interpolated electron and phonon bands.
3.  It then uses the output of this single `b2w` run to launch the `EpwA2fWorkChain` on different fine grids for a convergence test with respect to the Allen-Dynes Tc.
4.  It subsequently runs the `EpwIsoWorkChain` and `EpwAnisoWorkChain` to get the isotropic and anisotropic critical temperatures.

``transport.py``: `EpwTransportWorkChain`
=========================================
This is the workchain for calculating the transport properties.


``aiida_epw/controllers/``
*************************************
This module contains submission controllers based on `aiida-submission-controller`. The ``EpwSuperconWorkChainController`` provides a powerful interface for submitting large batches of ``EpwSuperconWorkChain`` calculations, for instance, for a high-throughput screening campaign across a group of structures. It handles duplicate checking and concurrency management. It is to be developed in the future.