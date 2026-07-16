"""Tests for the adaptive Eliashberg work chain."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory


class MockBuilder(dict):
    """Minimal process-builder stand-in for protocol tests."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def test_eliashberg_workchain_entry_point():
    """Test the workflow is registered through ``aiida.workflows``."""
    from aiida_epw.workflows.eliashberg import EliashbergWorkChain

    assert WorkflowFactory("epw.eliashberg") is EliashbergWorkChain
    assert "target" not in EliashbergWorkChain.spec().inputs
    assert "calculation_type" not in EliashbergWorkChain.spec().inputs
    assert "max_iterations" in EliashbergWorkChain.spec().inputs
    assert "max_sampling_iterations" not in EliashbergWorkChain.spec().inputs
    assert "result" in EliashbergWorkChain.spec().outputs
    assert "sampling_report" not in EliashbergWorkChain.spec().outputs
    assert hasattr(EliashbergWorkChain, "should_run_epw")
    assert not hasattr(EliashbergWorkChain, "should_run_sampling")
    assert not hasattr(EliashbergWorkChain, "analyze_sampling")


def test_setup_locks_calculation_type_to_eliashberg():
    """Test setup always submits EPW in Eliashberg mode."""
    from aiida_epw.common.types import CalculationTypes
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from aiida_epw.workflows.eliashberg import EliashbergWorkChain

    class FakeWorkChain:
        def __init__(self):
            self.ctx = AttributeDict()

        def exposed_inputs(self, workchain_class):
            assert workchain_class is EpwBaseWorkChain
            return AttributeDict(
                {
                    "parameters": orm.Dict(dict={"INPUTEPW": {"temps": [1.0, 2.0]}}),
                    "calculation_type": orm.EnumData(CalculationTypes.TRANSPORT),
                }
            )

    workchain = FakeWorkChain()

    EliashbergWorkChain.setup(workchain)

    assert (
        workchain.ctx.inputs.calculation_type.get_member()
        is CalculationTypes.ELIASHBERG
    )


def test_get_builder_from_protocol_locks_calculation_type(
    fixture_code, generate_structure, monkeypatch
):
    """Test protocol builder wraps the base builder without exposing calculation type."""
    from aiida_epw.common.types import CalculationTypes
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from aiida_epw.workflows.eliashberg import EliashbergWorkChain

    def mock_get_builder_from_protocol(**kwargs):
        builder = MockBuilder()
        builder.code = kwargs["code"]
        builder.structure = kwargs["structure"]
        builder.parameters = orm.Dict(dict={"INPUTEPW": {"temps": [1.0, 2.0]}})
        builder.calculation_type = orm.EnumData(CalculationTypes.TRANSPORT)
        return builder

    monkeypatch.setattr(
        EpwBaseWorkChain,
        "get_builder_from_protocol",
        mock_get_builder_from_protocol,
    )

    code = fixture_code("epw.epw")
    structure = generate_structure()
    builder = EliashbergWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol="fast",
        adaptive=False,
        max_iterations=2,
        sampling={"refine_points": 3},
    )

    assert builder.code is code
    assert builder.structure is structure
    assert "calculation_type" not in builder
    assert builder.adaptive.value is False
    assert builder.max_iterations.value == 2
    assert builder.sampling.get_dict() == {"refine_points": 3}


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
