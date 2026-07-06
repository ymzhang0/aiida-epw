"""Tests for EpwDegausswConvWorkChain."""

from unittest.mock import MagicMock
import numpy
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


def test_degaussw_get_builder_from_protocol_prep(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test get_builder_from_protocol for EpwDegausswConvWorkChain with EpwPrepWorkChain parent."""
    from aiida_epw.workflows.degaussw import EpwDegausswConvWorkChain
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

    builder = EpwDegausswConvWorkChain.get_builder_from_protocol(
        epw_code=epw_code,
        parent_epw=parent_epw,
        protocol="fast",
    )

    assert builder.structure is structure
    assert builder.parent_folder_epw is remote_stash


def test_degaussw_get_builder_from_protocol_base_success(
    fixture_code,
    generate_structure,
    generate_remote_data,
    fixture_localhost,
    monkeypatch,
):
    """Test get_builder_from_protocol for EpwDegausswConvWorkChain with EpwBaseWorkChain parent."""
    from aiida_epw.workflows.degaussw import EpwDegausswConvWorkChain
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

    builder = EpwDegausswConvWorkChain.get_builder_from_protocol(
        epw_code=epw_code,
        parent_epw=parent_epw,
        protocol="fast",
    )

    assert builder.structure is structure
    assert builder.parent_folder_epw is remote_stash


def test_degaussw_inspect_convergence():
    """Test inspect_convergence of EpwDegausswConvWorkChain using lambda."""
    from aiida_epw.workflows.degaussw import EpwDegausswConvWorkChain
    from types import SimpleNamespace
    from aiida.common.extendeddicts import AttributeDict

    class FakeExitCodes:
        ERROR_ALL_SUB_PROCESSES_FAILED = "ERROR_ALL_SUB_PROCESSES_FAILED"

    def make_mock_node(is_finished_ok, lambda_val):
        node = MagicMock()
        node.is_finished_ok = is_finished_ok
        node.exit_status = 0
        node.pk = 123
        node.outputs = MagicMock()
        node.outputs.a2f.get_lambda.return_value = numpy.array([lambda_val])
        node.outputs.output_parameters = orm.Dict(dict={})
        return node

    # degaussw = [0.05, 0.04, 0.03, 0.02, 0.01]
    # wc_0 (0.05) -> lambda = 1.5
    # wc_1 (0.04) -> lambda = 1.2 (diff vs 1.5 relative to 1.2 = 0.25)
    # wc_2 (0.03) -> lambda = 1.0 (diff vs 1.2 relative to 1.0 = 0.20)
    # wc_3 (0.02) -> lambda = 0.98 (diff vs 1.0 relative to 0.98 = 0.0204)
    # wc_4 (0.01) -> lambda = 0.97 (diff vs 0.98 relative to 0.97 = 0.0103)

    wc_0 = make_mock_node(True, 1.5)
    wc_1 = make_mock_node(True, 1.2)
    wc_2 = make_mock_node(True, 1.0)
    wc_3 = make_mock_node(True, 0.98)
    wc_4 = make_mock_node(True, 0.97)

    ctx = SimpleNamespace(
        degaussw_values=[0.05, 0.04, 0.03, 0.02, 0.01],
        wc_0=wc_0,
        wc_1=wc_1,
        wc_2=wc_2,
        wc_3=wc_3,
        wc_4=wc_4,
    )

    inputs = AttributeDict(
        {
            "convergence_threshold": orm.Float(0.05),
        }
    )

    reports = []
    wc_instance = SimpleNamespace(
        ctx=ctx,
        inputs=inputs,
        report=reports.append,
        exit_codes=FakeExitCodes(),
    )

    EpwDegausswConvWorkChain.inspect_convergence(wc_instance)

    # Should converge at index 3 (degaussw = 0.02 eV) because rel_diff (0.0204) <= threshold (0.05)
    assert wc_instance.ctx.converged_res["degaussw"] == 0.02
    assert wc_instance.ctx.converged_res["lambda"] == 0.98
    assert wc_instance.ctx.converged_res["workchain"] is wc_3
