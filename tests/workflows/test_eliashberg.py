"""Tests for the adaptive Eliashberg work chain."""

from unittest.mock import MagicMock

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
    assert "temps" in EliashbergWorkChain.spec().inputs
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
            self.inputs = AttributeDict()

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


def test_setup_uses_initial_temps_input():
    """Test explicit initial temperatures override protocol parameters."""
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from aiida_epw.workflows.eliashberg import EliashbergWorkChain

    class FakeWorkChain:
        def __init__(self):
            self.ctx = AttributeDict()
            self.inputs = AttributeDict({"temps": orm.List(list=[3.0, 1.0])})

        def exposed_inputs(self, workchain_class):
            assert workchain_class is EpwBaseWorkChain
            return AttributeDict(
                {
                    "parameters": orm.Dict(
                        dict={"INPUTEPW": {"temps": [10.0, 20.0], "nstemp": 2}}
                    ),
                }
            )

    workchain = FakeWorkChain()

    EliashbergWorkChain.setup(workchain)

    input_epw = workchain.ctx.inputs.parameters.get_dict()["INPUTEPW"]
    assert input_epw["temps"] == [1.0, 3.0]
    assert "nstemp" not in input_epw


def test_get_builder_from_protocol_uses_parent_epw(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test protocol builder starts from a parent Wannier-representation EPW run."""
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
    remote_stash = generate_remote_data(fixture_localhost, "/tmp/remote_stash")
    kpoints = orm.KpointsData()
    qpoints = orm.KpointsData()
    parent_epw = MagicMock()
    parent_epw.process_label = "EpwBaseWorkChain"
    parent_epw.inputs = MagicMock()
    parent_epw.inputs.structure = structure
    parent_epw.inputs.code = code
    parent_epw.inputs.kpoints = kpoints
    parent_epw.inputs.qpoints = qpoints
    parent_epw.outputs = MagicMock()
    parent_epw.outputs.remote_stash = remote_stash

    builder = EliashbergWorkChain.get_builder_from_protocol(
        epw_code=code,
        parent_epw=parent_epw,
        protocol="fast",
        adaptive=False,
        max_iterations=2,
        temps=[1.0, 2.0],
        sampling={"refine_points": 3},
    )

    assert builder.code is code
    assert builder.structure is structure
    assert builder.parent_folder_epw is remote_stash
    assert builder.kpoints is kpoints
    assert builder.qpoints is qpoints
    assert "calculation_type" not in builder
    assert builder.adaptive.value is False
    assert builder.max_iterations.value == 2
    assert builder.temps.get_list() == [1.0, 2.0]
    assert builder.sampling.get_dict() == {"refine_points": 3}


def test_get_builder_from_protocol_uses_eliashberg_protocol(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test workflow defaults are loaded from the Eliashberg protocol file."""
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from aiida_epw.workflows.eliashberg import EliashbergWorkChain

    def mock_get_builder_from_protocol(**kwargs):
        builder = MockBuilder()
        builder.code = kwargs["code"]
        builder.structure = kwargs["structure"]
        builder.parameters = orm.Dict(dict={"INPUTEPW": {}})
        return builder

    monkeypatch.setattr(
        EpwBaseWorkChain,
        "get_builder_from_protocol",
        mock_get_builder_from_protocol,
    )

    code = fixture_code("epw.epw")
    structure = generate_structure()
    parent_epw = MagicMock()
    parent_epw.process_label = "EpwBaseWorkChain"
    parent_epw.inputs = MagicMock()
    parent_epw.inputs.structure = structure
    parent_epw.inputs.code = code
    parent_epw.inputs.kpoints = orm.KpointsData()
    parent_epw.inputs.qpoints = orm.KpointsData()
    parent_epw.outputs = MagicMock()
    parent_epw.outputs.remote_stash = generate_remote_data(
        fixture_localhost, "/tmp/remote_stash"
    )

    builder = EliashbergWorkChain.get_builder_from_protocol(
        epw_code=code,
        parent_epw=parent_epw,
        protocol="fast",
    )

    assert builder.adaptive.value is True
    assert builder.max_iterations.value == 2
    assert builder.temps.get_list() == [1.0, 5.0]
    assert builder.sampling.get_dict() == {"refine_points": 5}


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
