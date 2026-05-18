.. _prep:

=======================
EpwPrepWorkChain
=======================

This work chain is used to compute the electron-phonon coupling on Wannier basis.

The workflow is an aggregation of the following work chains:

- Wannier90BandsWorkChain/Wannier90OptimizeWorkChain: wannierization
- PhBaseWorkChain: dvscf on coarse q-point grid
- EpwBaseWorkChain: transformation from coarse-grid Bloch basis to Wannier basis

For the Wannier namespace, ``EpwPrepWorkChain`` will accept the following inputs:

.. code-block:: python

        spec.expose_inputs(
            Wannier90OptimizeWorkChain,
            namespace='w90_bands',
            exclude=(
                'structure',
                'clean_workdir',
                'nscf.kpoints',
                'nscf.kpoints_distance'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.'
            }
        )

In ``EpwBaseWorkChain``, the kpoints MUST be integral multiples of the qpoints along each directions.
If one wants ``EpwPrepWorkChain`` to take over the Wannierization, one must allow it to set the kpoints according to the qpoints of the ``PhBaseWorkChain``.

For the ph_base namespace, ``EpwPrepWorkChain`` will accept the following inputs:

.. code-block:: python

        spec.expose_inputs(
            PhBaseWorkChain,
            namespace='ph_base',
            exclude=(
                'clean_workdir',
                'qpoints',
                'qpoints_distance',
                'ph.parent_folder'
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` that does the `ph.x` calculation.'
            }
        )

We want a unified interface for the Wannier90 and Phonon work chains so the specification of qpoints or qpoints_distance are moved to the topmost level.

Besides, the ph.parent_folder of the Phonon work chain is always provided by the ``PwBaseWorkChain`` inside the ``Wannier90OptimizeWorkChain``.
This means we must always restart from ``Wannier90OptimizeWorkChain`` as long as the workdir of ``Wannier90OptimizeWorkChain`` is cleaned.

For the epw namespace, ``EpwPrepWorkChain`` will accept the following inputs:


.. code-block:: python

        spec.expose_inputs(
            EpwBaseWorkChain,
            namespace='epw_base',
            exclude=(
                'structure',
                'clean_workdir',
                'qfpoints_distance',
                'qfpoints',
                'kfpoints',
                'kfpoints_factor',
            ),
            namespace_options={
                'required': False,
                'populate_defaults': False,
                'help': 'Inputs for the `EpwBaseWorkChain` that does the `epw.x` calculation.'
            }
        )

The ``EpwBaseWorkChain`` will not accept any inputs related to fine-grid settings.

Because we separate the Bloch-to-Wannier transformation from the interpolation, we set no interpolation of the find grids,

.. code-block:: text

    nkf1 = 1
    nkf2 = 1
    nkf3 = 1
    nqf1 = 1
    nqf2 = 1
    nqf3 = 1





.. code-block:: text

    wdata(1) = 'bands_plot = .true.'
    wdata(2) = 'begin kpoint_path'
    wdata(3) = 'G 0.00 0.00 0.00 M 0.50 0.00 0.00'
    wdata(4) = 'M 0.50 0.00 0.00 K 0.333333333333 0.333333333333 0.00'
    wdata(5) = 'K 0.333333333333 0.333333333333 0.00 G 0.0 0.0 0.00'
    wdata(6) = 'end kpoint_path'
    wdata(7) = 'bands_plot_format = gnuplot'
    wdata(8) = 'dis_num_iter      = 5000'
    wdata(9) = 'num_print_cycles  = 10'
    wdata(10) = 'dis_mix_ratio     = 1.0'
    wdata(11) = 'conv_tol = 1E-12'
    wdata(12) = 'conv_window = 4'


At the end of this step, you have the electron-phonon matrix element in real space
This needs to be stored for sure
The most important and only big file is "PREFIX..epmatwp"
For TiO, this is 872 Mb ...
From this file you can interpolate to any fine grid density

I mean use for next calculation but you may want to keep it in order to do more convergence later
for example if you find that a 40x40x40 grid is not enough
you dont want to redo step 1 and 2 to get 60x60x60

Files to save:

.. code-block:: text

    ln -s ../epw8-conv1/crystal.fmt
    ln -s ../epw8-conv1/epwdata.fmt
    ln -s ../epw8-conv1/<prefix>.bvec
    ln -s ../epw8-conv1/<prefix>.chk
    ln -s ../epw8-conv1/<prefix>.kgmap
    ln -s ../epw8-conv1/<prefix>.kmap
    ln -s ../epw8-conv1/<prefix>.mmn
    ln -s ../epw8-conv1/<prefix>.nnkp
    ln -s ../epw8-conv1/<prefix>.ukk
    ln -s ../epw8-conv1/<prefix>.epmatwp # Note: quite big file!
    ln -s ../epw8-conv1/vmedata.fmt
    ln -s ../epw8-conv1/dmedata.fmt
    ln -s ../epw8-conv1/save # Is basically the save folder from step 1

