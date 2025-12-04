---
title: Quick Start
---

This guide will help you get started quickly with `aiida-epw` for electron-phonon coupling calculations.

:::{note}
If you haven't installed `aiida-epw` yet, please refer to the [Installation Guide](installation.md) first.
:::

## Basic Concepts

`aiida-epw` provides the following main components:

- **`EpwCalculation`**: A calculation job that directly runs `epw.x` calculations
- **`EpwBaseWorkChain`**: A base workchain for running EPW calculations with automatic restart and error handling
- **`EpwPrepWorkChain`**: A complete preparation workchain that includes Wannier90, phonon, and EPW calculations
- **`SuperConWorkChain`**: A workchain for calculating superconducting critical temperature

## Usage Examples

### Example 1: Using EpwPrepWorkChain for Complete Calculation

`EpwPrepWorkChain` is the simplest approach, as it automatically handles all necessary steps:

```python
from aiida import orm, load_profile
from aiida_epw.workflows.prep import EpwPrepWorkChain
from aiida_wannier90_workflows.common.types import WannierProjectionType

load_profile()

# Prepare structure
structure = orm.StructureData(...)  # Your crystal structure

# Prepare code dictionary
codes = {
    'epw': orm.load_code('epw@localhost'),  # EPW code
    'pw': orm.load_code('pw@localhost'),    # PW code
    'ph': orm.load_code('ph@localhost'),    # PH code
    'wannier90': orm.load_code('w90@localhost'),  # Wannier90 code
}

# Create builder using protocol
builder = EpwPrepWorkChain.get_builder_from_protocol(
    codes=codes,
    structure=structure,
    protocol='moderate',
    wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
    reference_bands=reference_bands,  # Reference bands
    bands_kpoints=bands_kpoints,      # Band k-points
)

# Submit calculation
from aiida.engine import submit
workchain = submit(builder)
print(f"Submitted workchain: {workchain.pk}")
```

### Example 2: Using EpwBaseWorkChain

If you already have results from Wannier90 and phonon calculations, you can directly use `EpwBaseWorkChain`:

```python
from aiida import orm
from aiida_epw.workflows.base import EpwBaseWorkChain

# Prepare inputs
code = orm.load_code('epw@localhost')
structure = orm.StructureData(...)

# Get parent folders from previous calculations
parent_folder_nscf = ...  # RemoteData from nscf calculation
parent_folder_ph = ...    # RemoteData from ph calculation
parent_folder_chk = ...    # RemoteData from Wannier90 calculation

# Prepare k-points and q-points
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh([4, 4, 4])

qpoints = orm.KpointsData()
qpoints.set_kpoints_mesh([4, 4, 4])

# Create builder using protocol
builder = EpwBaseWorkChain.get_builder_from_protocol(
    code=code,
    structure=structure,
    protocol='moderate',
)

# Set parent folders
builder.parent_folder_nscf = parent_folder_nscf
builder.parent_folder_ph = parent_folder_ph
builder.parent_folder_chk = parent_folder_chk

# Set k-points and q-points
builder.kpoints = kpoints
builder.qpoints = qpoints

# Set fine grids (optional)
builder.qfpoints_distance = orm.Float(0.1)  # Fine q-point distance
builder.kfpoints_factor = orm.Int(2)        # Fine k-point factor

# Submit calculation
from aiida.engine import submit
workchain = submit(builder)
```

### Example 3: Calculating Superconducting Critical Temperature

Use `SuperConWorkChain` to calculate the superconducting critical temperature:

```python
from aiida import orm
from aiida_epw.workflows.supercon import SuperConWorkChain

# Get results from EpwPrepWorkChain
prep_workchain = orm.load_node(...)  # Previous EpwPrepWorkChain node

# Prepare code
epw_code = orm.load_code('epw@localhost')

# Create builder using protocol
builder = SuperConWorkChain.get_builder_from_protocol(
    epw_code=epw_code,
    parent_epw=prep_workchain,
    protocol='moderate',
    interpolation_distance=orm.Float(0.1),  # Interpolation distance
    convergence_threshold=orm.Float(0.1),   # Convergence threshold
)

# Submit calculation
from aiida.engine import submit
workchain = submit(builder)

# Get results
workchain = orm.load_node(workchain.pk)
if workchain.is_finished_ok:
    tc = workchain.outputs.Tc_iso.value
    print(f"Superconducting critical temperature: {tc} K")
```

### Example 4: Direct Use of EpwCalculation

For advanced users, you can directly use `EpwCalculation`:

```python
from aiida import orm
from aiida.plugins import CalculationFactory

EpwCalculation = CalculationFactory('epw.epw')

# Prepare inputs
builder = EpwCalculation.get_builder()
builder.code = orm.load_code('epw@localhost')
builder.structure = structure
builder.kpoints = kpoints
builder.qpoints = qpoints
builder.kfpoints = kfpoints
builder.qfpoints = qfpoints

# Set parameters
parameters = {
    'INPUTEPW': {
        'elph': True,
        'epwwrite': True,
        'epwread': False,
        'wannierize': False,
        'vme': 'dipole',
        'degaussw': 0.2,
        'degaussq': 0.05,
        'fsthick': 100.0,
        'temps': 300,
    }
}
builder.parameters = orm.Dict(parameters)

# Set parent folders
builder.parent_folder_nscf = parent_folder_nscf
builder.parent_folder_ph = parent_folder_ph
builder.parent_folder_chk = parent_folder_chk

# Set calculation options
builder.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 4}
builder.metadata.options.max_wallclock_seconds = 3600

# Submit calculation
from aiida.engine import submit
calc = submit(builder)
```

## Viewing Results

After the calculation is complete, you can view the results:

```python
# Load workchain node
workchain = orm.load_node(workchain_pk)

# Check status
if workchain.is_finished_ok:
    # Get output parameters
    output_params = workchain.outputs.output_parameters.get_dict()
    print("Output parameters:", output_params)
    
    # Get a2f data (if calculated)
    if 'a2f' in workchain.outputs:
        a2f = workchain.outputs.a2f
        frequency = a2f.get_array('frequency')
        a2f_value = a2f.get_array('a2f')
        print("Frequency range:", frequency.min(), "to", frequency.max())
    
    # Get max eigenvalue (if calculated)
    if 'max_eigenvalue' in workchain.outputs:
        max_eigenvalue = workchain.outputs.max_eigenvalue
        print("Max eigenvalue data calculated")
    
    # Get band structure (if calculated)
    if 'el_band_structure' in workchain.outputs:
        el_bands = workchain.outputs.el_band_structure
        print("Electronic band structure calculated")
    
    if 'ph_band_structure' in workchain.outputs:
        ph_bands = workchain.outputs.ph_band_structure
        print("Phonon band structure calculated")
else:
    print(f"Calculation failed, exit status: {workchain.exit_status}")
    print(f"Exit message: {workchain.exit_message}")
```

## Protocols

`aiida-epw` uses a protocol system to simplify input setup. Currently available protocols include:

- **`moderate`**: Medium-precision calculations that balance computational cost and accuracy

You can override the default values of a protocol using the `overrides` parameter:

```python
builder = EpwBaseWorkChain.get_builder_from_protocol(
    code=code,
    structure=structure,
    protocol='moderate',
    overrides={
        'parameters': {
            'INPUTEPW': {
                'degaussw': 0.1,  # Override default value
            }
        }
    }
)
```

## Frequently Asked Questions

### 1. How to choose appropriate k-point and q-point grids?

- **Coarse grid**: Typically use 4×4×4 to 8×8×8, depending on system size
- **Fine grid**: Usually 2-4 times the coarse grid, or use a smaller distance parameter

### 2. How to choose Wannier projection type?

- **`ATOMIC_PROJECTORS_QE`**: Uses Quantum ESPRESSO atomic projectors, requires reference bands
- **`SCDM`**: Uses the Selected Columns of the Density Matrix method

### 3. How long do calculations take?

EPW calculations typically take a long time, especially for:
- Large systems (many atoms)
- Dense k/q-point grids
- Fine grid interpolation

It is recommended to use appropriate computational resources and set a reasonable `max_wallclock_seconds`.

## Next Steps

- Check the [Developer Guide](developer.md) to learn how to contribute code
- Check the source code for more detailed API documentation
- Check the test files for more usage examples

## Getting Help

If you encounter problems:

1. Check AiiDA documentation: https://aiida.readthedocs.io/
2. Check EPW documentation: https://epw-code.org/
3. Submit an issue: https://github.com/aiidaplugins/aiida-epw/issues
