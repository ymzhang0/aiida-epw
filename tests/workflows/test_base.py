"""Tests for the `EpwBaseWorkChain` class and its error handlers."""

from unittest.mock import MagicMock
from aiida import orm

try:
    from aiida_epw.common.types import RestartType

    HAS_RESTART_TYPE = True
except ImportError:
    HAS_RESTART_TYPE = False
from aiida_epw.workflows.base import EpwBaseWorkChain


def test_handle_pade_approximants(aiida_localhost):
    """Test the handle_pade_approximants error handler with successful/failed temperatures and nsiw reduction."""

    class MockWorkChain:
        _MAX_NSIW = EpwBaseWorkChain._MAX_NSIW
        _MIN_NSIW = EpwBaseWorkChain._MIN_NSIW
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_pade_approximants = EpwBaseWorkChain.handle_pade_approximants

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock outputs: 2.0 succeeded, 1.0 failed
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "2.0": {
                    "nsiw": 400,
                    "iterations": {"ethr": [1e-8], "znormi": [1.2], "deltai": [2.5]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "1.0": {
                    "nsiw": 800,
                    "iterations": {"ethr": [None], "znormi": [None], "deltai": [None]},
                    "pade": {"delta": None, "znorm": None},
                },
            }
        }
    )

    # Setup ctx inputs
    initial_params = {"INPUTEPW": {"temps": [1.0, 2.0], "nstemp": 2}}
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    # Run handler
    report = workchain.handle_pade_approximants.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    input_epw = updated_params["INPUTEPW"]

    # Successfully calculated temperature 2.0 should be popped
    assert input_epw["temps"] == [1.0]
    assert input_epw["nstemp"] == 1

    # npade should be reduced to 25 to achieve N=200 with nsiw=800
    assert input_epw["npade"] == 25

    # Restart settings should be set
    if HAS_RESTART_TYPE:
        assert workchain.ctx.inputs.restart_type == RestartType.EPHREAD
    else:
        assert input_epw.get("epwread") is True
    assert workchain.ctx.inputs.parent_folder_epw == calc.outputs.remote_folder


def test_handle_pade_approximants_lower_bound(aiida_localhost):
    """Test that nsiw reduction respects the lower bound threshold (min limit)."""

    class MockWorkChain:
        _MAX_NSIW = EpwBaseWorkChain._MAX_NSIW
        _MIN_NSIW = EpwBaseWorkChain._MIN_NSIW
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_pade_approximants = EpwBaseWorkChain.handle_pade_approximants

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock output where nsiw is already small
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "1.0": {
                    "nsiw": 30,
                    "iterations": {"ethr": [None], "znormi": [None], "deltai": [None]},
                }
            }
        }
    )

    # Setup ctx inputs
    initial_params = {"INPUTEPW": {"temps": [1.0], "nstemp": 1}}
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    # Run handler
    report = workchain.handle_pade_approximants.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    # nsiw=30, current_N = 90*30/100 = 27. target_N = 20. npade = 20*100/30 = 66
    assert updated_params["INPUTEPW"]["npade"] == 66


def test_handle_pade_approximants_string_temps(aiida_localhost):
    """Test that the handler works correctly when temps is a space-separated string."""

    class MockWorkChain:
        _MAX_NSIW = EpwBaseWorkChain._MAX_NSIW
        _MIN_NSIW = EpwBaseWorkChain._MIN_NSIW
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_pade_approximants = EpwBaseWorkChain.handle_pade_approximants

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock outputs: 2.0 succeeded, 1.0 failed
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "2.0": {
                    "nsiw": 400,
                    "iterations": {"ethr": [1e-8], "znormi": [1.2], "deltai": [2.5]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "1.0": {
                    "nsiw": 800,
                    "iterations": {"ethr": [None], "znormi": [None], "deltai": [None]},
                    "pade": {"delta": None, "znorm": None},
                },
            }
        }
    )

    # Setup ctx inputs with space-separated string
    initial_params = {"INPUTEPW": {"temps": "1.0 2.0", "nstemp": 2}}
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    # Run handler
    report = workchain.handle_pade_approximants.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    input_epw = updated_params["INPUTEPW"]

    # Successfully calculated temperature 2.0 should be popped
    # And since the original input was a string, the result should also be a string
    assert input_epw["temps"] == "1.0"
    assert input_epw["nstemp"] == 1
    # npade should be reduced to 25 to achieve N=200 with nsiw=800
    assert input_epw["npade"] == 25


def test_handle_pade_approximants_linear_range(aiida_localhost):
    """Test that the handler correctly pops a linear range of temperatures."""

    class MockWorkChain:
        _MAX_NSIW = EpwBaseWorkChain._MAX_NSIW
        _MIN_NSIW = EpwBaseWorkChain._MIN_NSIW
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_pade_approximants = EpwBaseWorkChain.handle_pade_approximants

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock outputs: 1.0, 2.0, 3.0, 4.0 succeeded, 5.0 failed
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "1.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [1e-8]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "2.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [1e-8]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "3.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [1e-8]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "4.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [1e-8]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "5.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [None]},
                    "pade": {"delta": None, "znorm": None},
                },
            }
        }
    )

    # Setup ctx inputs as range from 1.0 to 10.0 with 10 values
    initial_params = {"INPUTEPW": {"temps": "1.0 10.0", "nstemp": 10}}
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    # Run handler
    report = workchain.handle_pade_approximants.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    input_epw = updated_params["INPUTEPW"]

    # 1.0, 2.0, 3.0, 4.0 are popped out. Remaining range starts at 5.0 and ends at 10.0
    # There are 6 temperatures remaining in the sequence (5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    assert input_epw["temps"] == "5.0 10.0"
    assert input_epw["nstemp"] == 6


def test_handle_pade_approximants_range_nstemp_2(aiida_localhost):
    """Test that the handler correctly processes linear range with nstemp=2."""

    class MockWorkChain:
        _MAX_NSIW = EpwBaseWorkChain._MAX_NSIW
        _MIN_NSIW = EpwBaseWorkChain._MIN_NSIW
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_pade_approximants = EpwBaseWorkChain.handle_pade_approximants

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock outputs: 1.0 succeeded, 10.0 failed
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "1.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [1e-8]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "10.0": {
                    "nsiw": 100,
                    "iterations": {"ethr": [None]},
                    "pade": {"delta": None, "znorm": None},
                },
            }
        }
    )

    # Setup ctx inputs: "1.0 10.0" with nstemp=2
    initial_params = {"INPUTEPW": {"temps": "1.0 10.0", "nstemp": 2}}
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    # Run handler
    report = workchain.handle_pade_approximants.__wrapped__(calc)

    assert report.do_break is True
    assert report.exit_code.status == 0

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    input_epw = updated_params["INPUTEPW"]

    # Since 1.0 succeeded and 10.0 failed, only 10.0 remains
    assert input_epw["temps"] == "10.0"
    assert input_epw["nstemp"] == 1


def test_handle_temperature_out_of_range(aiida_localhost):
    """Test the handle_temperature_out_of_range error handler."""

    class MockWorkChain:
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.ctx.is_finished = False
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_temperature_out_of_range = (
            EpwBaseWorkChain.handle_temperature_out_of_range
        )

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Run handler
    report = workchain.handle_temperature_out_of_range.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0
    assert workchain.ctx.is_finished is True
    assert any("phase transition" in msg for msg in workchain.report_messages)


def test_handle_out_of_walltime(aiida_localhost):
    """Test that handle_out_of_walltime works correctly when Eliashberg and EPHWRITE are configured."""

    class MockWorkChain:
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_out_of_walltime = EpwBaseWorkChain.handle_out_of_walltime

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Setup ctx inputs for Eliashberg + EPHWRITE
    initial_params = {
        "INPUTEPW": {
            "eliashberg": True,
            "epwread": True,
            "ephwrite": True,
        }
    }
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    if HAS_RESTART_TYPE:
        from aiida_epw.common.types import CalculationTypes, RestartType

        calc_type_mock = MagicMock()
        calc_type_mock.get_member.return_value = CalculationTypes.ELIASHBERG
        restart_type_mock = MagicMock()
        restart_type_mock.get_member.return_value = RestartType.EPHWRITE

        workchain.ctx.inputs.calculation_type = calc_type_mock
        workchain.ctx.inputs.restart_type = restart_type_mock
        workchain.ctx.inputs.__contains__.side_effect = lambda key: key in (
            "calculation_type",
            "restart_type",
            "parameters",
        )

    # Run handler
    report = workchain.handle_out_of_walltime.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0
    assert workchain.ctx.inputs.parent_folder_epw == calc.outputs.remote_folder

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    assert updated_params["INPUTEPW"]["restart"] is True


def test_handle_out_of_walltime_unsupported(aiida_localhost):
    """Test that handle_out_of_walltime aborts for unsupported calculation/restart configurations."""

    class MockWorkChain:
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_out_of_walltime = EpwBaseWorkChain.handle_out_of_walltime

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Setup ctx inputs for transport (not Eliashberg)
    initial_params = {
        "INPUTEPW": {
            "eliashberg": False,
            "scattering": True,
            "epwread": False,
            "ephwrite": True,
        }
    }
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    if HAS_RESTART_TYPE:
        from aiida_epw.common.types import CalculationTypes, RestartType

        calc_type_mock = MagicMock()
        calc_type_mock.get_member.return_value = CalculationTypes.TRANSPORT
        restart_type_mock = MagicMock()
        restart_type_mock.get_member.return_value = RestartType.EPHWRITE

        workchain.ctx.inputs.calculation_type = calc_type_mock
        workchain.ctx.inputs.restart_type = restart_type_mock
        workchain.ctx.inputs.__contains__.side_effect = lambda key: key in (
            "calculation_type",
            "restart_type",
            "parameters",
        )

    # Run handler
    report = workchain.handle_out_of_walltime.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert (
        report.exit_code.status
        == EpwBaseWorkChain.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE.status
    )


def test_handle_out_of_walltime_ephread_success(aiida_localhost):
    """Test that handle_out_of_walltime EPHREAD mode correctly pops succeeded temperatures and restarts."""

    class MockWorkChain:
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_out_of_walltime = EpwBaseWorkChain.handle_out_of_walltime

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock output parameters: 2.0 succeeded, 1.0 failed
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "2.0": {
                    "nsiw": 400,
                    "iterations": {"ethr": [1e-8], "znormi": [1.2], "deltai": [2.5]},
                    "pade": {"delta": 2.4, "znorm": 1.2},
                },
                "1.0": {
                    "nsiw": 800,
                    "iterations": {"ethr": [None], "znormi": [None], "deltai": [None]},
                    "pade": {"delta": None, "znorm": None},
                },
            }
        }
    )

    # Setup ctx inputs for Eliashberg + EPHREAD
    initial_params = {
        "INPUTEPW": {
            "eliashberg": True,
            "epwread": True,
            "ephwrite": False,
            "temps": [1.0, 2.0],
            "nstemp": 2,
        }
    }
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    if HAS_RESTART_TYPE:
        from aiida_epw.common.types import CalculationTypes, RestartType

        calc_type_mock = MagicMock()
        calc_type_mock.get_member.return_value = CalculationTypes.ELIASHBERG
        restart_type_mock = MagicMock()
        restart_type_mock.get_member.return_value = RestartType.EPHREAD

        workchain.ctx.inputs.calculation_type = calc_type_mock
        workchain.ctx.inputs.restart_type = restart_type_mock
        workchain.ctx.inputs.__contains__.side_effect = lambda key: key in (
            "calculation_type",
            "restart_type",
            "parameters",
        )

    # Run handler
    report = workchain.handle_out_of_walltime.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert report.exit_code.status == 0
    assert workchain.ctx.inputs.parent_folder_epw == calc.outputs.remote_folder

    updated_params = workchain.ctx.inputs.parameters.get_dict()
    # Successfully calculated temperature 2.0 should be popped, 1.0 remains
    assert updated_params["INPUTEPW"]["temps"] == [1.0]
    assert updated_params["INPUTEPW"]["nstemp"] == 1


def test_handle_out_of_walltime_ephread_failure(aiida_localhost):
    """Test that handle_out_of_walltime EPHREAD mode aborts if no temperatures succeeded."""

    class MockWorkChain:
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_out_of_walltime = EpwBaseWorkChain.handle_out_of_walltime

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Mock output parameters: no temperatures succeeded
    calc.outputs.output_parameters = orm.Dict(
        dict={
            "isotropic_eliashberg": {
                "1.0": {
                    "nsiw": 800,
                    "iterations": {"ethr": [None], "znormi": [None], "deltai": [None]},
                    "pade": {"delta": None, "znorm": None},
                },
            }
        }
    )

    # Setup ctx inputs for Eliashberg + EPHREAD
    initial_params = {
        "INPUTEPW": {
            "eliashberg": True,
            "epwread": True,
            "ephwrite": False,
            "temps": [1.0],
            "nstemp": 1,
        }
    }
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    if HAS_RESTART_TYPE:
        from aiida_epw.common.types import CalculationTypes, RestartType

        calc_type_mock = MagicMock()
        calc_type_mock.get_member.return_value = CalculationTypes.ELIASHBERG
        restart_type_mock = MagicMock()
        restart_type_mock.get_member.return_value = RestartType.EPHREAD

        workchain.ctx.inputs.calculation_type = calc_type_mock
        workchain.ctx.inputs.restart_type = restart_type_mock
        workchain.ctx.inputs.__contains__.side_effect = lambda key: key in (
            "calculation_type",
            "restart_type",
            "parameters",
        )

    # Run handler
    report = workchain.handle_out_of_walltime.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert (
        report.exit_code.status
        == EpwBaseWorkChain.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE.status
    )


def test_handle_out_of_walltime_wannierize(aiida_localhost):
    """Test that handle_out_of_walltime Wannierize mode prints a message and aborts."""

    class MockWorkChain:
        exit_codes = EpwBaseWorkChain.exit_codes

        def __init__(self):
            self.ctx = MagicMock()
            self.report_messages = []

        def report(self, msg):
            self.report_messages.append(msg)

        def report_error_handled(self, calculation, action):
            self.report(f"Calculation failed: {action}")

        handle_out_of_walltime = EpwBaseWorkChain.handle_out_of_walltime

    workchain = MockWorkChain()

    # Mock calculation
    calc = MagicMock()
    calc.outputs = MagicMock()
    calc.outputs.remote_folder = orm.RemoteData(
        computer=aiida_localhost, remote_path="/tmp"
    )

    # Setup ctx inputs for Eliashberg + WANNIERIZE
    initial_params = {
        "INPUTEPW": {
            "eliashberg": True,
            "wannierize": True,
            "epwread": False,
        }
    }
    workchain.ctx.inputs = MagicMock()
    workchain.ctx.inputs.parameters = orm.Dict(dict=initial_params)

    if HAS_RESTART_TYPE:
        from aiida_epw.common.types import CalculationTypes, RestartType

        calc_type_mock = MagicMock()
        calc_type_mock.get_member.return_value = CalculationTypes.ELIASHBERG
        restart_type_mock = MagicMock()
        restart_type_mock.get_member.return_value = RestartType.WANNIERIZE

        workchain.ctx.inputs.calculation_type = calc_type_mock
        workchain.ctx.inputs.restart_type = restart_type_mock
        workchain.ctx.inputs.__contains__.side_effect = lambda key: key in (
            "calculation_type",
            "restart_type",
            "parameters",
        )

    # Run handler
    report = workchain.handle_out_of_walltime.__wrapped__(calc)

    # Assertions
    assert report.do_break is True
    assert (
        report.exit_code.status
        == EpwBaseWorkChain.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE.status
    )
    assert any(
        "Resuming Wannierization via epbread is not implemented yet" in msg
        for msg in workchain.report_messages
    )
