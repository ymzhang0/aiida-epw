"""Base work chain for the EPW calculations."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    process_handler,
    while_,
)
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_epw.tools.kpoints import check_kpoints_qpoints_compatibility
from aiida_epw.tools.workchain import (
    find_related_calculation,
    get_parent_folder_calculation,
    get_parent_ph_qpoints,
    validate_parent_ph_inputs,
)

EpwCalculation = CalculationFactory("epw.epw")


def get_kpoints_from_chk_folder(chk_folder):
    """This method tries different strategies to find the k-point mesh from a parent nscf folder.

    :param chk_folder: A RemoteData node from a Wannier90Calculation (chk).
    :return: A KpointsData node that has mesh information.
    :raises ValueError: If the mesh cannot be found through any strategy.
    """
    wannier_params = chk_folder.creator.inputs.parameters

    if "mp_grid" in wannier_params:
        mp_grid = wannier_params["mp_grid"]
        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh(mp_grid)
        return kpoints
    else:
        raise ValueError(
            "Could not deduce mesh from the parent folder of the nscf calculation."
        )


def derive_inputepw_parameters(
    parameters, parent_folder_chk=None, parent_folder_epw=None
):
    """Return EPW parameters updated from Wannier/EPW parent folders."""
    parameters = parameters.copy()
    inputepw = parameters.setdefault("INPUTEPW", {})

    if parent_folder_chk is not None:
        w90_params = parent_folder_chk.creator.inputs.parameters.get_dict()
        exclude_bands = w90_params.get("exclude_bands", None)

        if exclude_bands:
            inputepw["bands_skipped"] = (
                f"exclude_bands = {exclude_bands[0]}:{exclude_bands[-1]}"
            )

        inputepw["nbndsub"] = w90_params["num_wann"]

    if parent_folder_epw is not None:
        calculation = find_related_calculation(parent_folder_epw)
        epw_params = calculation.inputs.parameters.get_dict()["INPUTEPW"]

        inputepw["use_ws"] = epw_params.get("use_ws", False)
        inputepw["nbndsub"] = epw_params["nbndsub"]
        if "bands_skipped" in epw_params:
            inputepw["bands_skipped"] = epw_params.get("bands_skipped")

    return parameters


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the entire input namespace of `EpwBaseWorkChain`."""
    # Usually at the creation of the inputs, the coarse and fine k/q grid is already determined.
    #
    # Cannot specify both `kfpoints` and `kfpoints_factor`
    if all(_ in inputs for _ in ["kfpoints", "kfpoints_factor"]):
        return "Can only specify one of the `kfpoints`, `kfpoints_factor`."

    if not any([_ in inputs for _ in ["kfpoints", "kfpoints_factor"]]):
        return "Either `kfpoints` or `kfpoints_factor` must be specified."

    # Cannot specify both `kfpoints` and `kfpoints_factor`
    if all(_ in inputs for _ in ["qfpoints", "qfpoints_distance"]):
        return "Can only specify one of the `qfpoints`, `qfpoints_distance`."

    if not any([_ in inputs for _ in ["qfpoints", "qfpoints_distance"]]):
        return "Either `qfpoints` or `qfpoints_distance` must be specified."

    return None


class EpwBaseWorkChain(ProtocolMixin, BaseRestartWorkChain):
    """BaseWorkchain to run a epw.x calculation."""

    _process_class = EpwCalculation

    defaults = AttributeDict(
        {
            "min_num_mpiprocs_per_machine_for_oom_restart": 32,
        }
    )

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        # Here we exclude the `metadata` input of the EpwCalculation
        # to avoid the conflict between it with the `metadata` of this WorkChain.
        # We also exclude the `qfpoints` and `kfpoints` inputs of the EpwCalculation
        # because they are marked as required in the EpwCalculation.
        # but will only be provided when the EpwBaseWorkChain is run.
        spec.expose_inputs(
            EpwCalculation,
            exclude=('metadata', 'parent_folder', 'qfpoints', 'kfpoints', 'kpoints', 'qpoints')
        )

        spec.input(
            'options',
            valid_type=orm.Dict,
            required=True,
            serializer=to_aiida_type,
            help=(
                "The options dictionary for the calculation."
                "It must be defined in the top-level as a solution of the conflict between `metadata_calculation` of the WorkChain and `metadata` of the Calculation."
                )
            )

        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help=(
                "The structure data to use for the generation of k/q points by `create_kpoints_from_distance` calcfunction."
                "In principle, we should take the structure as the one we used in the previous calculation."
                "However, it is a bit difficult to take all the restart cases into account if we have a long chain of EPW calculations."
                "Therefore, for now we just provide it manually as an input."
                "But in the future, it will be removed."
                "In cases that the coarse and fine k/q points are explicitly speficied, this input is not necessary anymore."
                )
            )

        spec.input(
            'qfpoints_distance',
            valid_type=orm.Float,
            serializer=to_aiida_type,
            required=False,
            help=(
                "The q-points distance to generate the find qpoints"
                "If specified, the fine qpoints will be generated from `create_kpoints_from_distance` calcfunction."
                "If not specified, the fine qpoints will be read from the inputs.qfpoints input."
                )
            )


        spec.input(
            'kfpoints_factor',
            valid_type=orm.Int,
            serializer=to_aiida_type,
            required=False,
            help=(
                "The factor to multiply the q-point mesh to get the fine k-point mesh"
                "If not specified, the fine kpoints will be generated from the parent folder of the nscf calculation."
                )
            )

        spec.input(
            'w90_chk_to_ukk_script',
            valid_type=orm.RemoteData,
            required=False,
            help=(
                "A julia script to convert the prefix.chk file (generated by wannier90.x) "
                "to a prefix.ukk file (to be used by epw.x). "
                "If provided along with parent_folder_chk, the script will be prepended to the batch script."
            )
        )

        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,
            cls.validate_kpoints,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(EpwCalculation)

        spec.exit_code(202, 'ERROR_COARSE_GRID_NOT_VALID',
            message='The specification of coarse k/q grid is not valid.')
        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unidentified unrecoverable error.')
        spec.exit_code(310, 'ERROR_KNOWN_UNRECOVERABLE_FAILURE',
            message='The calculation failed with a known unrecoverable error.')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "base.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        protocol=None,
        overrides=None,
        options=None,
        w90_chk_to_ukk_script=None,
        **_,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.epw`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param w90_chk_to_ukk_script: a julia script to convert the prefix.chk file (generated by wannier90.x) to a prefix.ukk file (to be used by epw.x)
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import (
            recursive_merge,
        )

        type_check(code, orm.Code)
        type_check(structure, orm.StructureData)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        # Update the parameters based on the protocol inputs
        parameters = inputs["parameters"]

        # If overrides are provided, they are considered absolute
        if overrides:
            parameter_overrides = overrides.get("parameters", {})
            parameters = recursive_merge(parameters, parameter_overrides)

        # pylint: disable=no-member
        builder = cls.get_builder()
        builder.structure = structure
        builder.code = code
        builder.parameters = orm.Dict(parameters)
        ## Must firstly pop the options from the metadata dictionary.
        builder.options = orm.Dict(inputs.pop("options"))

        if w90_chk_to_ukk_script:
            type_check(w90_chk_to_ukk_script, orm.RemoteData)
            builder.w90_chk_to_ukk_script = w90_chk_to_ukk_script

        if "settings" in inputs:
            builder.settings = orm.Dict(inputs["settings"])
        if "parallelization" in inputs:
            builder.parallelization = orm.Dict(inputs["parallelization"])

        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])

        builder.qfpoints_distance = orm.Float(inputs["qfpoints_distance"])
        builder.kfpoints_factor = orm.Int(inputs["kfpoints_factor"])
        builder.max_iterations = orm.Int(inputs["max_iterations"])
        # pylint: enable=no-member

        return builder

    def setup(self):
        """Call the ``setup`` of the ``BaseRestartWorkChain`` and create the inputs dictionary in ``self.ctx.inputs``.
        This ``self.ctx.inputs`` dictionary will be used by the ``EpwCalculation`` in the internal loop.
        """
        super().setup()

        self.ctx.inputs = AttributeDict(self.exposed_inputs(EpwCalculation))

        # Initialize here an empty metadata dictionary.
        # Now there is only options in it.

        # TODO: Should check if we need to append more
        # information into it.
        metadata = {}

        metadata["options"] = self.inputs.options.get_dict()

        self.ctx.inputs.metadata = metadata

        # Persist derived parameters on the workchain inputs so restart provenance
        # reflects the actual values submitted to the calcjob.
        parameters = derive_inputepw_parameters(
            self.ctx.inputs.parameters.get_dict(),
            parent_folder_chk=self.inputs.get("parent_folder_chk"),
            parent_folder_epw=self.inputs.get("parent_folder_epw"),
        )
        self.ctx.inputs.parameters = orm.Dict(parameters)

    # We should validate the kpoints and qpoints on the fly
    # because they are usually not determined at the creation of the inputs.
    def validate_kpoints(self):
        """Validate the inputs related to k-points.
        `epw.x` requires coarse k-points and q-points to be compatible, which means the kpoints should be multiple of qpoints.
        e.g. if qpoints are [2,2,2], kpoints should be [2*l,2*m,2*n] for integer l,m,n.
        We firstly construct qpoints. Either an explicit `KpointsData` with given mesh/path, or a desired qpoints distance should be specified.
        In the case of the latter, the `KpointsData` will be constructed for the input `StructureData` using the `create_kpoints_from_distance` calculation function.
        Then we construct kpoints by multiplying the qpoints mesh by the `kpoints_factor`.
        """
        # If there is already the parent folder of a previous EPW calculation, the coarse k/q grid is already there and must be valid.
        # We only need to take the kpointsdata from it and continue to generate the find grid.

        if "parent_folder_epw" in self.inputs:
            epw_calc = find_related_calculation(self.inputs.parent_folder_epw)
            kpoints = epw_calc.inputs.kpoints
            qpoints = epw_calc.inputs.qpoints

        # If there is no parent folder of a previous EPW calculation, it must be the case that we are running the transition from coarse BLoch representation to Wannier representation.
        # This means that we are using coarse k grid from a previous nscf calculation and
        else:
            if "kpoints" in self.inputs:
                kpoints = self.inputs.kpoints
            elif "parent_folder_nscf" in self.inputs:
                kpoints = get_parent_folder_calculation(
                    self.inputs.parent_folder_nscf
                ).inputs.kpoints
            elif "parent_folder_chk" in self.inputs:
                kpoints = get_kpoints_from_chk_folder(self.inputs.parent_folder_chk)
            else:
                self.report(
                    "Could not determine the coarse k-points from the inputs, the parent folder of the NSCF calculation, or the parent folder of the wannier90 calculation."
                )
                return self.exit_codes.ERROR_COARSE_GRID_NOT_VALID

            if "qpoints" in self.inputs:
                qpoints = self.inputs.qpoints
            elif "parent_folder_ph" in self.inputs:
                if "structure" in self.inputs:
                    qpoints = validate_parent_ph_inputs(
                        self.inputs.parent_folder_ph, self.inputs.structure
                    )
                else:
                    qpoints = get_parent_ph_qpoints(self.inputs.parent_folder_ph)
            else:
                self.report(
                    "Could not determine the coarse q-points from the inputs or the parent folder of the ph calculation."
                )
                return self.exit_codes.ERROR_COARSE_GRID_NOT_VALID

        self.report(
            f"Successfully determined coarse k-points from the inputs: {kpoints.get_kpoints_mesh()[0]}"
        )
        self.report(
            f"Successfully determined coarse q-points from the inputs: {qpoints.get_kpoints_mesh()[0]}"
        )

        is_compatible, message = check_kpoints_qpoints_compatibility(kpoints, qpoints)

        self.ctx.inputs.kpoints = kpoints
        self.ctx.inputs.qpoints = qpoints

        if not is_compatible:
            self.report(message)
            return self.exit_codes.ERROR_COARSE_GRID_NOT_VALID

        ## TODO: If we are restarting from .ephmat folder, we should use the same
        ## qfpoints and kfpoints as the creator of 'parent_folder_epw'.
        if "qfpoints" in self.inputs:
            qfpoints = self.inputs.qfpoints
        else:
            inputs = {
                "structure": self.inputs.structure,
                "distance": self.inputs.qfpoints_distance,
                "force_parity": self.inputs.get(
                    "qfpoints_force_parity", orm.Bool(False)
                ),
                "metadata": {"call_link_label": "create_qfpoints_from_distance"},
            }
            qfpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg

        if "kfpoints" in self.inputs:
            kfpoints = self.inputs.kfpoints
        else:
            qfpoints_mesh = qfpoints.get_kpoints_mesh()[0]
            kfpoints = orm.KpointsData()
            kfpoints.set_kpoints_mesh(
                [v * self.inputs.kfpoints_factor.value for v in qfpoints_mesh]
            )

        self.ctx.inputs.qfpoints = qfpoints
        self.ctx.inputs.kfpoints = kfpoints

    def set_restart_from_calculation(self, calculation):
        """Update the next EPW submission to restart from a previous calculation."""
        remote_folder = getattr(getattr(calculation, "outputs", None), "remote_folder", None)
        if remote_folder is None:
            return False

        parameters = self.ctx.inputs.parameters.get_dict()
        inputepw = parameters.setdefault("INPUTEPW", {})

        self.ctx.inputs.parent_folder_epw = remote_folder

        if inputepw.get("elph", False):
            inputepw["epwread"] = True

        if inputepw.get("eliashberg", False) and inputepw.get("ephwrite", True):
            inputepw["restart"] = True

        self.ctx.inputs.parameters = orm.Dict(parameters)
        return True

    def prepare_process(self):
        """Prepare inputs for the next calculation.

        If a previous calculation was marked as a clean EPW restart point, update
        the inputs so the next iteration restarts from its remote folder.
        """
        restart_calc = getattr(self.ctx, "restart_calc", None)
        if restart_calc is None:
            return None

        if not self.set_restart_from_calculation(restart_calc):
            self.report(
                f"Could not determine a remote folder for {restart_calc.process_label}<{restart_calc.pk}>."
            )
            return self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE

        self.ctx.restart_calc = None
        return None

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [
            calculation.process_label,
            calculation.pk,
            calculation.exit_status,
            calculation.exit_message,
        ]
        self.report("{}<{}> failed with exit status {}: {}".format(*arguments))
        self.report(f"Action taken: {action}")

    @process_handler(
        priority=600,
        exit_codes=[EpwCalculation.exit_codes.ERROR_MEMORY_EXCEEDS_MAX_MEMLT],
    )
    def handle_known_unrecoverable_failure(self, calculation):
        """Abort failures that are known to be unrecoverable."""
        action = "known unrecoverable failure detected, aborting..."
        self.report_error_handled(calculation, action)
        return ProcessHandlerReport(
            True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE
        )

    @process_handler(
        priority=610,
        exit_codes=EpwCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME,
    )
    def handle_scheduler_out_of_walltime(self, calculation):
        """Attempt a best-effort restart after a scheduler walltime stop.

        Unlike `ERROR_OUT_OF_WALLTIME`, a scheduler-enforced stop does not guarantee
        that EPW wrote a clean restart point. However, on many systems EPW still
        leaves a usable restart folder behind, so retry from the latest remote
        folder before giving up.
        """
        if not self.set_restart_from_calculation(calculation):
            action = (
                "scheduler walltime detected but no remote folder is available, "
                "aborting..."
            )
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE
            )

        self.ctx.restart_calc = calculation
        action = (
            "attempting a restart from the previous EPW remote folder."
        )
        self.report_error_handled(calculation, action)
        return ProcessHandlerReport(True)

    @process_handler(
        priority=615,
        exit_codes=EpwCalculation.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY,
    )
    def handle_scheduler_out_of_memory(self, calculation):
        """Abort scheduler OOM failures without attempting a restart."""
        action = (
            "scheduler out-of-memory detected; EPW restarts are not considered "
            "safe after changing the MPI layout, aborting..."
        )
        self.report_error_handled(calculation, action)
        return ProcessHandlerReport(
            True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE
        )

    @process_handler(
        priority=605,
        exit_codes=EpwCalculation.exit_codes.ERROR_OUT_OF_WALLTIME,
    )
    def handle_out_of_walltime(self, calculation):
        """Restart clean walltime exits from the latest EPW remote folder."""
        if not self.set_restart_from_calculation(calculation):
            action = "clean walltime exit detected but no remote folder is available, aborting..."
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE
            )

        self.ctx.restart_calc = calculation
        action = "clean walltime exit detected, restarting from the latest EPW remote folder."
        self.report_error_handled(calculation, action)
        return ProcessHandlerReport(True)

    @process_handler(
        priority=620,
        exit_codes=[EpwCalculation.exit_codes.ERROR_CANNOT_BRACKET_EF],
    )
    def handle_cannot_bracket_ef(self, calculation):
        """Handle the `cannot bracket Ef` error by explicitly setting the Fermi energy.
        
        This handler retrieves the Fermi energy from the parent NSCF calculation
        and injects it into the INPUTEPW parameters for the restart.
        """
        action = "Cannot bracket Fermi energy. Setting `fermi_energy` manually from parent calculation and restarting..."
        self.report_error_handled(calculation, action)

        # 1. 提取当前的输入参数
        parameters = self.ctx.inputs.parameters.get_dict()

        # 2. 尝试从父代计算（NSCF）中提取 Fermi Energy
        fermi_energy = None
        if "parent_folder_nscf" in self.inputs:
            try:
                # 追溯到生成 nscf remote folder 的原计算节点
                nscf_calc = get_parent_folder_calculation(self.inputs.parent_folder_nscf)
                # 从 NSCF 的输出参数字典中读取 fermi_energy (单位是 eV)
                fermi_energy = nscf_calc.outputs.output_parameters.dict.fermi_energy
            except (AttributeError, KeyError):
                self.report("Could not extract `fermi_energy` from parent NSCF outputs. Aborting...")
                return ProcessHandlerReport(True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE)
        else:
            self.report("No `parent_folder_nscf` found in inputs to extract `fermi_energy`. Aborting...")
            return ProcessHandlerReport(True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE)

        if fermi_energy is None:
            self.report("The extracted `fermi_energy` is None. Aborting...")
            return ProcessHandlerReport(True, self.exit_codes.ERROR_KNOWN_UNRECOVERABLE_FAILURE)

        self.report(f"Extracted Fermi energy: {fermi_energy} eV. Updating parameters...")

        # 3. 将 fermi_energy 注入到 EPW 参数中
        parameters["INPUTEPW"]["fermi_energy"] = fermi_energy
        self.ctx.inputs.parameters = orm.Dict(parameters)

        # 4. 指示 WorkChain 使用更新后的参数自动重启
        return ProcessHandlerReport(True)

    @process_handler(priority=500)
    def handle_unrecoverable_failure(self, calculation):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if calculation.is_failed and calculation.exit_status < 400:
            self.report_error_handled(calculation, "unrecoverable error, aborting...")
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE
            )
