.. _overview:

===================
Project Structure
===================

This page summarizes the parts of ``aiida-epw`` that are implemented in the current repository.

High-Level Overview
-------------------

.. code-block:: text

   aiida-epw/
   ├── src/
   │   └── aiida_epw/
   │       ├── calculations/
   │       │   └── epw.py
   │       ├── parsers/
   │       │   └── epw.py
   │       ├── tools/
   │       │   ├── calculators.py
   │       │   ├── kpoints.py
   │       │   ├── parsers.py
   │       │   ├── plot.py
   │       │   └── workchain.py
   │       └── workflows/
   │           ├── base.py
   │           ├── prep.py
   │           └── supercon.py
   ├── docs/
   ├── tests/
   └── pyproject.toml

Module Breakdown
----------------

``calculations/epw.py``
   Defines ``EpwCalculation``, the calcjob wrapper for ``epw.x`` input preparation, file staging, and retrieval.

``parsers/epw.py``
   Defines ``EpwParser``, which turns retrieved EPW outputs into AiiDA data nodes such as ``Dict``, ``XyData``, and ``BandsData``.

``workflows/base.py``
   Defines ``EpwBaseWorkChain``, the restart-capable wrapper around a single ``EpwCalculation``.

``workflows/prep.py``
   Defines ``EpwPrepWorkChain``, which orchestrates Wannier90, phonon, and coarse-to-Wannier EPW preparation steps.

``workflows/supercon.py``
   Defines ``SuperConWorkChain``, which runs interpolation and final Eliashberg calculations for superconductivity workflows.

``tools/``
   Contains helper utilities for post-processing, k-point validation, plotting, and workflow restart bookkeeping.
