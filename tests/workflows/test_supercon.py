"""Tests for SuperConWorkChain."""

from unittest.mock import MagicMock
import pytest
from aiida import orm


class MockBuilder(dict):
    def __getattr__(self, key):
        if key == "get_dict":
            return lambda: {}
        if key == "get_list":
            return lambda: []
        return self.setdefault(key, MockBuilder())

    def __setattr__(self, key, value):
        self[key] = value

    def __call__(self, *args, **kwargs):
        return self


def mock_get_builder(*args, **kwargs):
    return MockBuilder()


def test_supercon_get_builder_from_protocol_prep(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test get_builder_from_protocol for SuperConWorkChain with EpwPrepWorkChain parent."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from plumpy.ports import Port, PortNamespace

    monkeypatch.setattr(Port, "validate", lambda *a, **k: None)
    monkeypatch.setattr(PortNamespace, "validate", lambda *a, **k: None)
    monkeypatch.setattr(EpwBaseWorkChain, "get_builder_from_protocol", mock_get_builder)

    epw_code = fixture_code("epw.epw")
    structure = generate_structure()
    remote_stash = generate_remote_data(fixture_localhost, "/tmp/remote_stash")

    # Mock the EpwPrepWorkChain parent node
    parent_epw = MagicMock()
    parent_epw.process_label = "EpwPrepWorkChain"
    parent_epw.inputs = MagicMock()
    parent_epw.inputs.structure = structure

    # Mock base.links.get_outgoing
    epw_source = MagicMock()
    epw_source.inputs = MagicMock()
    epw_source.inputs.code = epw_code
    epw_source.inputs.kpoints = orm.KpointsData()
    epw_source.inputs.qpoints = orm.KpointsData()
    epw_source.outputs = MagicMock()
    epw_source.outputs.remote_stash = remote_stash

    parent_epw.base.links.get_outgoing.return_value.first.return_value.node = epw_source

    builder = SuperConWorkChain.get_builder_from_protocol(
        epw_code=epw_code,
        parent_epw=parent_epw,
        protocol="fast",
    )

    assert builder.structure is structure
    assert builder.parent_folder_epw is remote_stash


def test_supercon_get_builder_from_protocol_base_success(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test get_builder_from_protocol for SuperConWorkChain with EpwBaseWorkChain parent having structure."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from plumpy.ports import Port, PortNamespace

    monkeypatch.setattr(Port, "validate", lambda *a, **k: None)
    monkeypatch.setattr(PortNamespace, "validate", lambda *a, **k: None)
    monkeypatch.setattr(EpwBaseWorkChain, "get_builder_from_protocol", mock_get_builder)

    epw_code = fixture_code("epw.epw")
    structure = generate_structure()
    remote_stash = generate_remote_data(fixture_localhost, "/tmp/remote_stash")

    # Mock the EpwBaseWorkChain parent node
    parent_epw = MagicMock()
    parent_epw.process_label = "EpwBaseWorkChain"
    parent_epw.inputs = MagicMock()
    parent_epw.inputs.structure = structure
    parent_epw.inputs.code = epw_code
    parent_epw.inputs.kpoints = orm.KpointsData()
    parent_epw.inputs.qpoints = orm.KpointsData()
    parent_epw.outputs = MagicMock()
    parent_epw.outputs.remote_stash = remote_stash

    builder = SuperConWorkChain.get_builder_from_protocol(
        epw_code=epw_code,
        parent_epw=parent_epw,
        protocol="fast",
    )

    assert builder.structure is structure
    assert builder.parent_folder_epw is remote_stash


def test_supercon_get_builder_from_protocol_base_failure(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test get_builder_from_protocol for SuperConWorkChain with EpwBaseWorkChain parent missing structure raises ValueError."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from plumpy.ports import Port, PortNamespace

    monkeypatch.setattr(Port, "validate", lambda *a, **k: None)
    monkeypatch.setattr(PortNamespace, "validate", lambda *a, **k: None)
    monkeypatch.setattr(EpwBaseWorkChain, "get_builder_from_protocol", mock_get_builder)

    epw_code = fixture_code("epw.epw")
    remote_stash = generate_remote_data(fixture_localhost, "/tmp/remote_stash")

    # Mock the EpwBaseWorkChain parent node without structure
    parent_epw = MagicMock()
    parent_epw.process_label = "EpwBaseWorkChain"
    parent_epw.inputs = MagicMock()
    # Delete structure attribute to raise AttributeError when accessed
    del parent_epw.inputs.structure
    parent_epw.inputs.code = epw_code
    parent_epw.inputs.kpoints = orm.KpointsData()
    parent_epw.inputs.qpoints = orm.KpointsData()
    parent_epw.outputs = MagicMock()
    parent_epw.outputs.remote_stash = remote_stash

    with pytest.raises(ValueError, match="does not contain `structure` in its inputs"):
        SuperConWorkChain.get_builder_from_protocol(
            epw_code=epw_code,
            parent_epw=parent_epw,
            protocol="fast",
        )


def test_supercon_should_run_final():
    """Test should_run_final with different states of epw_interp and convergence."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from types import SimpleNamespace
    from aiida.common.extendeddicts import AttributeDict

    # Mock exit codes
    class FakeExitCodes:
        ERROR_SUB_PROCESS_EPW_INTERP = "ERROR_SUB_PROCESS_EPW_INTERP"
        ERROR_ALLEN_DYNES_NOT_CONVERGED = "ERROR_ALLEN_DYNES_NOT_CONVERGED"

    # Define fake workchain helper
    def make_fake_workchain(epw_interp_list, is_converged, always_run_final):
        reports = []
        ctx = SimpleNamespace(
            epw_interp=epw_interp_list,
            is_converged=is_converged,
        )
        inputs = AttributeDict(
            {
                "always_run_final": SimpleNamespace(value=always_run_final),
            }
        )
        return SimpleNamespace(
            inputs=inputs,
            ctx=ctx,
            exit_codes=FakeExitCodes(),
            report=reports.append,
            reports=reports,
        )

    # 1. epw_interp is empty -> should return ERROR_SUB_PROCESS_EPW_INTERP
    wc1 = make_fake_workchain(
        epw_interp_list=[], is_converged=True, always_run_final=True
    )
    assert SuperConWorkChain.should_run_final(wc1) == "ERROR_SUB_PROCESS_EPW_INTERP"
    assert "empty" in wc1.reports[0]

    # 2. epw_interp is not empty, converged -> should return True
    wc2 = make_fake_workchain(
        epw_interp_list=[object()], is_converged=True, always_run_final=False
    )
    assert SuperConWorkChain.should_run_final(wc2) is True

    # 3. epw_interp is not empty, not converged but always_run_final is True -> should return True
    wc3 = make_fake_workchain(
        epw_interp_list=[object()], is_converged=False, always_run_final=True
    )
    assert SuperConWorkChain.should_run_final(wc3) is True

    # 4. epw_interp is not empty, not converged, always_run_final is False -> should return ERROR_ALLEN_DYNES_NOT_CONVERGED
    wc4 = make_fake_workchain(
        epw_interp_list=[object()], is_converged=False, always_run_final=False
    )
    assert SuperConWorkChain.should_run_final(wc4) == "ERROR_ALLEN_DYNES_NOT_CONVERGED"


def test_epw_base_restart_types(fixture_code, generate_structure):
    """Test that EpwBaseWorkChain exposes restart_type from EpwCalculation."""
    from aiida_epw.workflows.base import EpwBaseWorkChain
    from aiida_epw.common import RestartType

    epw_code = fixture_code("epw.epw")
    structure = generate_structure()

    builder = EpwBaseWorkChain.get_builder_from_protocol(
        code=epw_code,
        structure=structure,
        protocol="fast",
    )

    # We should be able to set and access restart_type on the builder
    builder.restart_type = RestartType.EPHWRITE
    assert builder.restart_type == RestartType.EPHWRITE


def test_supercon_get_builder_from_protocol_default(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test get_builder_from_protocol default values."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from plumpy.ports import Port, PortNamespace

    monkeypatch.setattr(Port, "validate", lambda *a, **k: None)
    monkeypatch.setattr(PortNamespace, "validate", lambda *a, **k: None)

    epw_code = fixture_code("epw.epw")
    structure = generate_structure()
    remote_stash = generate_remote_data(fixture_localhost, "/tmp/remote_stash")

    parent_epw = MagicMock()
    parent_epw.process_label = "EpwBaseWorkChain"
    parent_epw.inputs = MagicMock()
    parent_epw.inputs.structure = structure
    parent_epw.inputs.code = epw_code
    parent_epw.inputs.kpoints = orm.KpointsData()
    parent_epw.inputs.qpoints = orm.KpointsData()
    parent_epw.outputs = MagicMock()
    parent_epw.outputs.remote_stash = remote_stash

    builder = SuperConWorkChain.get_builder_from_protocol(
        epw_code=epw_code,
        parent_epw=parent_epw,
        protocol="fast",
    )

    # Verify that the builders are configured according to the protocol
    assert "code" in builder.epw_interp
    assert "code" in builder.epw_final_iso
    assert "code" in builder.epw_final_aniso

    # epw_interp check
    if "restart_type" in builder.epw_interp:
        from aiida_epw.common import RestartType

        assert builder.epw_interp.restart_type == RestartType.EPHWRITE

    # epw_final_iso check
    if "momentum_dependence" in builder.epw_final_iso:
        assert not builder.epw_final_iso.momentum_dependence.value
        assert not builder.epw_final_iso.real_axis.value
    if "calculation_type" in builder.epw_final_iso:
        from aiida_epw.common.types import CalculationTypes

        assert (
            builder.epw_final_iso.calculation_type.get_member()
            == CalculationTypes.ELIASHBERG
        )
    if "restart_type" in builder.epw_final_iso:
        from aiida_epw.common import RestartType

        assert builder.epw_final_iso.restart_type == RestartType.EPHREAD
    assert (
        builder.epw_final_iso.parameters.get_dict()["INPUTEPW"].get("tc_linear", False)
        is False
    )

    # epw_final_aniso check
    if "momentum_dependence" in builder.epw_final_aniso:
        assert builder.epw_final_aniso.momentum_dependence.value
        assert not builder.epw_final_aniso.full_bandwidth.value
        assert not builder.epw_final_aniso.real_axis.value
    if "calculation_type" in builder.epw_final_aniso:
        from aiida_epw.common.types import CalculationTypes

        assert (
            builder.epw_final_aniso.calculation_type.get_member()
            == CalculationTypes.ELIASHBERG
        )
    if "restart_type" in builder.epw_final_aniso:
        from aiida_epw.common import RestartType

        assert builder.epw_final_aniso.restart_type == RestartType.EPHREAD


def test_supercon_results():
    """Test that results method attaches all available outputs correctly."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from types import SimpleNamespace

    # Mock outputs
    a2f_node = orm.Dict({"a2f": True})
    params_node = orm.Dict({"output_parameters": True})
    iso_gap_node = orm.Dict({"iso_gap": True})
    aniso_gap0_node = orm.Dict({"aniso_gap0": True})
    aniso_fs_node = orm.Dict({"aniso_fs": True})

    iso_outputs = SimpleNamespace(
        a2f=a2f_node,
        output_parameters=params_node,
        iso_gap_data=iso_gap_node,
    )
    aniso_outputs = SimpleNamespace(
        aniso_gap0_data=aniso_gap0_node,
        aniso_gap_FS=aniso_fs_node,
    )

    wc = MagicMock(spec=SuperConWorkChain)
    wc.ctx = SimpleNamespace(
        final_epw_iso=SimpleNamespace(outputs=iso_outputs),
        final_epw_aniso=SimpleNamespace(outputs=aniso_outputs),
        epw_interp=[],
    )
    registered_outputs = {}
    wc.out = lambda key, val: registered_outputs.update({key: val})

    SuperConWorkChain.results(wc)

    assert registered_outputs["a2f"] is a2f_node
    assert registered_outputs["parameters"] is params_node
    assert registered_outputs["iso_gap_data"] is iso_gap_node
    assert registered_outputs["aniso_gap0_data"] is aniso_gap0_node
    assert registered_outputs["aniso_gap_FS"] is aniso_fs_node
    assert "aniso_gap_imag" not in registered_outputs


def test_supercon_results_fallback_to_interp():
    """Test that results method falls back to epw_interp for a2f when final iso lacks a2f."""
    from aiida_epw.workflows.supercon import SuperConWorkChain
    from types import SimpleNamespace

    a2f_interp_node = orm.Dict({"interp_a2f": True})
    interp_outputs = SimpleNamespace(a2f=a2f_interp_node)

    wc = MagicMock(spec=SuperConWorkChain)
    wc.ctx = SimpleNamespace(
        epw_interp=[SimpleNamespace(outputs=interp_outputs)],
    )
    registered_outputs = {}
    wc.out = lambda key, val: registered_outputs.update({key: val})

    SuperConWorkChain.results(wc)

    assert registered_outputs["a2f"] is a2f_interp_node
