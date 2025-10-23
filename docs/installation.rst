.. _installation:

================
Installation
================

As a prerequisite, you need to have the following packages installed:

- aiida-core
- aiida-wannier90-workflows
- aiida-quantumespresso

To use aiida-epw as a broker to your EPW jobs, I strongly recommend you to install the newest version of quantum ESPRESSO 7.5 (with EPW 6.0).

Early versions of EPW are not well supported.

If you want to use the PDWF wannierization scheme, you need to install a specific branch of wannier90.

You may find it here: https://github.com/qiaojunfeng/wannier90/tree/p2w_ham

To facilitate conversion of .chk file to .ukk file, you need to install WannierIO.jl.

You may find it here: https://github.com/qiaojunfeng/wannier90/tree/p2w_ham

The package is not packed in PyPI. You can install it from the source code.

.. code-block:: bash

   git clone https://github.com/aiidaplugins/aiida-epw
   cd aiida-epw
   pip install (-e) .

