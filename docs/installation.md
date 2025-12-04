---
title: Installation
---

## Installation

First, make sure you have AiiDA and related dependencies installed. Then install `aiida-epw`:

```bash
pip install aiida-epw
```

Or install from source:

```bash
git clone https://github.com/aiidaplugins/aiida-epw.git
cd aiida-epw
pip install -e .
```

## Prerequisites

### Codes

Before using `aiida-epw`, you need to make sure Quantum ESPRESSO and EPW packages are successfully installed. More precisely, one should have `pw.x`, `ph.x`, `pw2wannier90.x`, `wannier90.x`, `epw.x` binarys in the cluster where the calculations are executed.

### Configuration

In the host where AiiDA engine is installed, one should set up the codes above using 

```bash
verdi code create core.code.installed --computer 'computer label' --filepath-executable 'filepath' --label 'code label'
```

Or you may configure it using python code

```python
from aiida import orm
from aiida.plugins import CodeFactory

CodeFactory = CodeFactory('epw.epw')
code = CodeFactory(
    label='epw',
    description='EPW code',
    input_plugin='epw.epw',
    remote_computer_exec=(computer, '/path/to/epw.x')
)
code.store()
```

### Configure pseudopotentials

Prepare the corresponding pseudopotential files

```bash
aiida-pseudo install 'pseudo family' -v 'version' -x 'exchange-correlation functional'
```