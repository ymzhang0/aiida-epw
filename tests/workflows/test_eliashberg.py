"""Tests for the adaptive Eliashberg work chain."""

from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory


def test_eliashberg_workchain_entry_point():
    """Test the workflow is registered through ``aiida.workflows``."""
    from aiida_epw.workflows.eliashberg import EliashbergWorkChain

    assert WorkflowFactory("epw.eliashberg") is EliashbergWorkChain
    assert "target" not in EliashbergWorkChain.spec().inputs


def test_extract_gap_series_infers_isotropic_from_inputs():
    """Test isotropic gap extraction is inferred from ``momentum_dependence``."""
    from aiida_epw.workflows.eliashberg import extract_gap_series

    class IsoGap:
        def get_gap_FS(self):
            return {"T": [1.0], "gap": [1.0]}

    outputs = AttributeDict({"iso_gap_functions": IsoGap()})

    assert extract_gap_series(outputs, is_anisotropic=False) == {
        "T": [1.0],
        "gap": [1.0],
    }


def test_extract_gap_series_infers_anisotropic_from_inputs():
    """Test anisotropic gap extraction is inferred from ``momentum_dependence``."""
    from aiida_epw.workflows.eliashberg import extract_gap_series

    class AnisoGap:
        def get_averaged_gap(self):
            return {"T": [1.0], "gap": [[1.0, 2.0]]}

    outputs = AttributeDict({"aniso_gap_functions": AnisoGap()})

    assert extract_gap_series(outputs, is_anisotropic=True) == {
        "T": [1.0],
        "gap": [[1.0, 2.0]],
    }
