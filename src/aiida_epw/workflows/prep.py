"""Work chain for doing the coarse-grid calculations."""

from pathlib import Path

from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.datastructures import StashMode
from aiida.engine import ToContext, WorkChain, if_
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
    create_kpoints_from_distance,
)
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.utils.workflows.builder.setter import (
    set_kpoints,
)
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90OptimizeWorkChain,
)
from aiida_wannier90_workflows.workflows.bands import (
    validate_inputs as validate_inputs_bands,
)

from aiida_epw.tools.workchain import (
    format_subprocess_failure,
    get_parent_folder_calculation,
    get_target_basepath,
)
from aiida_epw.workflows.base import EpwBaseWorkChain


def _ensure_stash_options(options, computer):
    """Normalize stash options to the schema expected by current aiida-core."""
    stash = options.get("stash")
    if not stash:
        return

    stash.setdefault("stash_mode", StashMode.COPY.value)
    stash.setdefault("target_base", get_target_basepath(computer))


def _as_mapping(value):
    """Return an AiiDA/Python mapping as a plain dictionary."""
    if isinstance(value, orm.Dict):
        return value.get_dict()
    return value or {}


def should_epw_wannierize(inputs) -> bool:
    """Return whether the EPW namespace is configured to run Wannierization directly."""
    epw_base = _as_mapping(inputs.get("epw_base"))
    parameters = _as_mapping(epw_base.get("parameters"))
    inputepw = _as_mapping(parameters.get("INPUTEPW", parameters.get("inputepw")))
    return bool(inputepw.get("wannierize", False))


def validate_inputs(  # pylint: disable=unused-argument,inconsistent-return-statements
    inputs, ctx=None
):
    """Validate the inputs of the `EpwPrepWorkChain`."""
    has_w90_bands = "w90_bands" in inputs
    use_epw_wannierize = should_epw_wannierize(inputs)

    if has_w90_bands and use_epw_wannierize:
        return (
            "`w90_bands` inputs and `epw_base.parameters.INPUTEPW.wannierize = True` "
            "are mutually exclusive."
        )

    if not has_w90_bands and not use_epw_wannierize:
        return (
            "Either provide `w90_bands` inputs or set "
            "`epw_base.parameters.INPUTEPW.wannierize = True`."
        )

    if use_epw_wannierize:
        missing = [namespace for namespace in ("scf", "nscf") if namespace not in inputs]
        if missing:
            return (
                "`scf` and `nscf` inputs are required when "
                "`epw_base.parameters.INPUTEPW.wannierize = True`."
            )


class EpwPrepWorkChain(ProtocolMixin, WorkChain):
    """Main work chain to start calculating properties using EPW.

    Has support for both the selected columns of the density matrix (SCDM) and
    (projectability-disentangled Wannier function) PDWF projection types.
    """

    @classmethod
    def define(cls, spec):
        """Define the work chain specification."""
        super().define(spec)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
        )
        spec.input(
            "qpoints_distance",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.5),
        )
        spec.input(
            "kpoints_distance_scf",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.15),
        )
        spec.input(
            "kpoints_factor_nscf",
            valid_type=orm.Int,
            default=lambda: orm.Int(2),
        )

        spec.input(
            "parent_folder_ph",
            valid_type=(orm.RemoteData, orm.RemoteStashFolderData),
            required=False,
        )

        spec.expose_inputs(
            Wannier90OptimizeWorkChain,
            namespace="w90_bands",
            exclude=(
                "structure",
                "clean_workdir",
            ),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the `Wannier90OptimizeWorkChain/Wannier90BandsWorkChain`.",
            },
        )
        spec.inputs["w90_bands"].validator = validate_inputs_bands
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=(
                "clean_workdir",
                "pw.structure",
                "kpoints",
                "kpoints_distance",
            ),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the SCF `PwBaseWorkChain` used by direct EPW Wannierization.",
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="nscf",
            exclude=(
                "clean_workdir",
                "pw.structure",
                "pw.parent_folder",
                "kpoints",
                "kpoints_distance",
            ),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Inputs for the NSCF `PwBaseWorkChain` used by direct EPW Wannierization.",
            },
        )
        spec.expose_inputs(
            PhBaseWorkChain,
            namespace="ph_base",
            exclude=(
                "clean_workdir",
                "ph.parent_folder",
                "qpoints",
                "qpoints_distance",
            ),
            namespace_options={
                "help": "Inputs for the `PhBaseWorkChain` that does the `ph.x` calculation."
            },
        )
        spec.expose_inputs(
            EpwBaseWorkChain,
            namespace="epw_base",
            exclude=(
                "structure",
                "clean_workdir",
                "kpoints",
                "qpoints",
                "kfpoints",
                "qfpoints",
                "qfpoints_distance",
                "kfpoints_factor",
                "parent_folder_ph",
                "parent_folder_nscf",
                "parent_folder_epw",
                "parent_folder_chk",
            ),
            namespace_options={"help": "Inputs for the `EpwBaseWorkChain`."},
        )
        spec.expose_inputs(
            EpwBaseWorkChain,
            namespace="epw_bands",
            exclude=(
                "structure",
                "clean_workdir",
                "kpoints",
                "qpoints",
                "kfpoints",
                "qfpoints",
                "qfpoints_distance",
                "kfpoints_factor",
                "parent_folder_ph",
                "parent_folder_nscf",
                "parent_folder_epw",
                "parent_folder_chk",
            ),
            namespace_options={
                "required": False,
                "help": "Inputs for the `EpwBaseWorkChain`.",
            },
        )
        spec.inputs.validator = validate_inputs
        spec.output("retrieved", valid_type=orm.FolderData)
        spec.output("epw_folder", valid_type=orm.RemoteStashFolderData)

        spec.outline(
            cls.generate_reciprocal_points,
            if_(cls.should_run_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            if_(cls.should_run_nscf)(
                cls.run_nscf,
                cls.inspect_nscf,
            ),
            if_(cls.should_run_wannier90)(
                cls.run_wannier90,
                cls.inspect_wannier90,
            ),
            cls.run_ph,
            cls.inspect_ph,
            cls.run_epw,
            cls.inspect_epw,
            if_(cls.should_run_epw_bands)(
                cls.run_epw_bands,
                cls.inspect_epw_bands,
            ),
            cls.results,
        )
        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="The SCF `PwBaseWorkChain` sub process failed",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_NSCF",
            message="The NSCF `PwBaseWorkChain` sub process failed",
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_PHONON",
            message="The electron-phonon `PhBaseWorkChain` sub process failed",
        )
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90",
            message="The `Wannier90BandsWorkChain` sub process failed",
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_EPW",
            message="The `EpwWorkChain` sub process failed",
        )
        spec.exit_code(
            406,
            "ERROR_SUB_PROCESS_FAILED_EPW_BANDS",
            message="The `EpwBandsWorkChain` sub process failed",
        )

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "prep.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        codes,
        structure,
        protocol=None,
        overrides=None,
        wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        reference_bands=None,
        bands_kpoints=None,
        parent_folder_ph=None,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()
        builder.structure = structure

        pseudo_family = inputs.pop("pseudo_family", None)
        w90_bands_inputs = inputs.get("w90_bands", {})
        use_epw_wannierize = should_epw_wannierize(inputs)

        if use_epw_wannierize:
            scf_inputs = recursive_merge(
                {"pseudo_family": pseudo_family},
                inputs.get("scf", {}),
            )
            scf = PwBaseWorkChain.get_builder_from_protocol(
                code=codes["pw"],
                structure=structure,
                protocol=protocol,
                overrides=scf_inputs,
                **kwargs,
            )
            scf.pop("clean_workdir", None)
            scf.pop("kpoints_distance", None)
            builder.scf = scf

            nscf_inputs = recursive_merge(
                {"pseudo_family": pseudo_family},
                inputs.get("nscf", {}),
            )
            nscf = PwBaseWorkChain.get_builder_from_protocol(
                code=codes["pw"],
                structure=structure,
                protocol=protocol,
                overrides=nscf_inputs,
                **kwargs,
            )
            nscf.pop("clean_workdir", None)
            nscf.pop("kpoints_distance", None)
            builder.nscf = nscf
            builder.pop("w90_bands", None)
        else:
            if reference_bands:
                w90_bands = Wannier90OptimizeWorkChain.get_builder_from_protocol(
                    structure=structure,
                    codes=codes,
                    pseudo_family=pseudo_family,
                    overrides=w90_bands_inputs,
                    projection_type=wannier_projection_type,
                    reference_bands=reference_bands,
                    bands_kpoints=bands_kpoints,
                )
                w90_bands.separate_plotting = False
            else:
                w90_bands = Wannier90BandsWorkChain.get_builder_from_protocol(
                    structure=structure,
                    codes=codes,
                    pseudo_family=pseudo_family,
                    overrides=w90_bands_inputs,
                    projection_type=wannier_projection_type,
                    bands_kpoints=bands_kpoints,
                )

            if wannier_projection_type == WannierProjectionType.ATOMIC_PROJECTORS_QE:
                w90_bands.pop("projwfc", None)

            w90_bands.pop("structure", None)
            w90_bands.pop("open_grid", None)

            builder.w90_bands = w90_bands
            builder.pop("scf", None)
            builder.pop("nscf", None)

        args = (codes["ph"], None, protocol)
        ph_base_inputs = inputs.get("ph_base", None)
        _ensure_stash_options(ph_base_inputs["ph"]["metadata"]["options"], codes["ph"].computer)
        ph_base = PhBaseWorkChain.get_builder_from_protocol(
            *args, overrides=ph_base_inputs, **kwargs
        )
        ph_base.pop("clean_workdir", None)
        ph_base.pop("qpoints_distance")

        builder.ph_base = ph_base

        # TODO:
        # Here I have a loop for the epw builders for furture extension of another epw bands interpolation
        # .
        for namespace in ["epw_base", "epw_bands"]:

            epw_inputs = inputs.get(namespace, None)
            if namespace == "epw_base":
                _ensure_stash_options(epw_inputs["options"], codes["epw"].computer)

            epw_builder = EpwBaseWorkChain.get_builder_from_protocol(
                code=codes["epw"],
                structure=structure,
                protocol=protocol,
                overrides=epw_inputs,
                **kwargs,
            )

            if "settings" in epw_inputs:
                epw_builder.settings = orm.Dict(epw_inputs["settings"])
            if "parallelization" in epw_inputs:
                epw_builder.parallelization = orm.Dict(epw_inputs["parallelization"])
            builder[namespace] = epw_builder

        builder.qpoints_distance = orm.Float(inputs["qpoints_distance"])
        builder.kpoints_distance_scf = orm.Float(inputs["kpoints_distance_scf"])
        builder.kpoints_factor_nscf = orm.Int(inputs["kpoints_factor_nscf"])
        if parent_folder_ph:
            builder.parent_folder_ph = parent_folder_ph
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])

        return builder

    def generate_reciprocal_points(self):
        """Generate the qpoints and kpoints meshes for the `ph.x` and `pw.x` calculations."""
        from aiida_wannier90_workflows.utils.kpoints import (
            get_explicit_kpoints
        )
        parent_folder_ph_calculation = None
        if "parent_folder_ph" in self.inputs:
            parent_folder_ph_calculation = get_parent_folder_calculation(
                self.inputs.parent_folder_ph
            )

        if (
            parent_folder_ph_calculation is not None
            and parent_folder_ph_calculation.process_label == "PhCalculation"
        ):
            qpoints = parent_folder_ph_calculation.inputs.qpoints
        else:
            inputs = {
                "structure": self.inputs.structure,
                "distance": self.inputs.qpoints_distance,
                "force_parity": self.inputs.get(
                    "kpoints_force_parity", orm.Bool(False)
                ),
                "metadata": {"call_link_label": "create_qpoints_from_distance"},
            }
            qpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg
        self.ctx.qpoints = qpoints

        if should_epw_wannierize(self.inputs):
            inputs = {
                "structure": self.inputs.structure,
                "distance": self.inputs.kpoints_distance_scf,
                "force_parity": self.inputs.get(
                    "kpoints_force_parity", orm.Bool(False)
                ),
                "metadata": {"call_link_label": "create_kpoints_scf_from_distance"},
            }
            kpoints_scf = create_kpoints_from_distance(**inputs)

            qpoints_mesh = qpoints.get_kpoints_mesh()[0]
            kpoints_nscf = orm.KpointsData()
            kpoints_nscf.set_kpoints_mesh(
                [v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh]
            )
        elif "w90_bands" in self.inputs:
            inputs = {
                "structure": self.inputs.structure,
                "distance": self.inputs.kpoints_distance_scf,
                "force_parity": self.inputs.get(
                    "kpoints_force_parity", orm.Bool(False)
                ),
                "metadata": {"call_link_label": "create_kpoints_scf_from_distance"},
            }

            kpoints_scf = create_kpoints_from_distance(**inputs)

            qpoints_mesh = qpoints.get_kpoints_mesh()[0]

            kpoints_nscf = orm.KpointsData()
            kpoints_nscf.set_kpoints_mesh(
                [v * self.inputs.kpoints_factor_nscf.value for v in qpoints_mesh]
            )
        
        self.ctx.kpoints_scf = kpoints_scf
        self.ctx.kpoints_nscf = kpoints_nscf

    def should_run_wannier90(self):
        """Check if the wannier90 workflow should be run."""
        return "w90_bands" in self.inputs and not should_epw_wannierize(self.inputs)

    def should_run_scf(self):
        """Check if the standalone SCF workflow should be run."""
        return should_epw_wannierize(self.inputs)

    def should_run_nscf(self):
        """Check if the standalone NSCF workflow should be run."""
        return should_epw_wannierize(self.inputs)

    def run_scf(self):
        """Run the standalone SCF workflow for direct EPW Wannierization."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="scf"))
        inputs.metadata.call_link_label = "scf"
        inputs.pw.structure = self.inputs.structure
        inputs.kpoints = self.ctx.kpoints_scf

        workchain_node = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching PwBaseWorkChain<{workchain_node.pk}> for SCF")

        return ToContext(workchain_scf=workchain_node)

    def inspect_scf(self):
        """Verify that the standalone SCF workflow finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(format_subprocess_failure(workchain, "PwBaseWorkChain"))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.parent_folder_scf = workchain.outputs.remote_folder

    def run_nscf(self):
        """Run the standalone NSCF workflow for direct EPW Wannierization."""
        from aiida_wannier90_workflows.utils.kpoints import (
            get_explicit_kpoints
        )
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="nscf"))
        inputs.metadata.call_link_label = "nscf"
        inputs.pw.structure = self.inputs.structure
        inputs.pw.parent_folder = self.ctx.workchain_scf.outputs.remote_folder
        inputs.kpoints = get_explicit_kpoints(self.ctx.kpoints_nscf)

        workchain_node = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"launching PwBaseWorkChain<{workchain_node.pk}> for NSCF")

        return ToContext(workchain_nscf=workchain_node)

    def inspect_nscf(self):
        """Verify that the standalone NSCF workflow finished successfully."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(format_subprocess_failure(workchain, "PwBaseWorkChain"))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.parent_folder_nscf = workchain.outputs.remote_folder

    def run_wannier90(self):
        """Run the wannier90 workflow."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace="w90_bands")
        )
        if "reference_bands" in self.inputs.w90_bands:
            w90_class = Wannier90OptimizeWorkChain
            # inputs.pop('projwfc')
        else:
            w90_class = Wannier90BandsWorkChain

        self.ctx.w90_class_name = w90_class.get_name()
        self.report(f"Running a {self.ctx.w90_class_name}.")

        inputs.metadata.call_link_label = "w90_bands"
        inputs.structure = self.inputs.structure

        set_kpoints(inputs, self.ctx.kpoints_nscf, w90_class)
        inputs["scf"]["kpoints"] = self.ctx.kpoints_scf

        workchain_node = self.submit(w90_class, **inputs)
        self.report(f"launching {w90_class.get_name()}<{workchain_node.pk}>")

        return ToContext(workchain_w90_bands=workchain_node)

    def inspect_wannier90(self):
        """Verify that the wannier90 workflow finished successfully."""
        workchain = self.ctx.workchain_w90_bands

        if not workchain.is_finished_ok:
            self.report(format_subprocess_failure(workchain, self.ctx.w90_class_name))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        scf_base_wc = (
            workchain.base.links.get_outgoing(link_label_filter="scf").first().node
        )
        self.ctx.parent_folder_scf = scf_base_wc.outputs.remote_folder

        nscf_base_wc = (
            workchain.base.links.get_outgoing(link_label_filter="nscf").first().node
        )
        self.ctx.parent_folder_nscf = nscf_base_wc.outputs.remote_folder

        optimize_disproj = getattr(workchain.inputs, "optimize_disproj", False)
        if (
            self.ctx.w90_class_name == "Wannier90OptimizeWorkChain"
            and optimize_disproj
        ):
            self.ctx.parent_folder_chk = (
                workchain.outputs.wannier90_optimal__remote_folder
            )
        else:
            self.ctx.parent_folder_chk = workchain.outputs.wannier90.remote_folder

    def run_ph(self):
        """Run the `PhBaseWorkChain`."""
        inputs = AttributeDict(
            self.exposed_inputs(PhBaseWorkChain, namespace="ph_base")
        )

        parent_folder_ph_calculation = None
        if (
            "parent_folder_ph" in self.inputs
            and (
                parent_folder_ph_calculation := get_parent_folder_calculation(
                    self.inputs.parent_folder_ph
                )
            ).process_label
            == "PhCalculation"
        ):
            inputs.ph.parent_folder = self.inputs.parent_folder_ph
            inputs.ph.qpoints = parent_folder_ph_calculation.inputs.qpoints
        else:
            inputs.ph.parent_folder = self.ctx.parent_folder_scf

        inputs.qpoints = self.ctx.qpoints

        inputs.metadata.call_link_label = "ph_base"
        workchain_node = self.submit(PhBaseWorkChain, **inputs)
        self.report(f"launching PhBaseWorkChain<{workchain_node.pk}>")

        return ToContext(workchain_ph=workchain_node)

    def inspect_ph(self):
        """Verify that the `PhBaseWorkChain` finished successfully."""
        workchain = self.ctx.workchain_ph

        if not workchain.is_finished_ok:
            self.report(format_subprocess_failure(workchain, "PhBaseWorkChain"))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PHONON

        self.ctx.parent_folder_ph = workchain.outputs.remote_folder

    def run_epw(self):
        """Run the `EpwBaseWorkChain`."""
        inputs = AttributeDict(
            self.exposed_inputs(EpwBaseWorkChain, namespace="epw_base")
        )

        inputs.structure = self.inputs.structure

        # The EpwBaseWorkChain will take the parent folder of the previous
        # PhCalculation, PwCalculation, and Wannier90Calculation.
        inputs.parent_folder_ph = self.ctx.parent_folder_ph
        inputs.parent_folder_nscf = self.ctx.parent_folder_nscf
        if "parent_folder_chk" in self.ctx:
            inputs.parent_folder_chk = self.ctx.parent_folder_chk

        fine_points = orm.KpointsData()
        fine_points.set_kpoints_mesh([1, 1, 1])

        # Here we explicitely speficy the coarse k/q grid so EpwBaseWorkChain will not deduce it from
        # the parent folders.
        # This EpwBaseWorkChain is only used for the transition from coarse BLoch representation to Wannier representation.
        # Thus the find grid is always [1, 1, 1].
        inputs.kpoints = self.ctx.kpoints_nscf
        inputs.kfpoints = fine_points
        inputs.qpoints = self.ctx.qpoints
        inputs.qfpoints = fine_points

        # The update of epw parameters according to the wannier parameters
        # and the file copying and conversion
        # is now handled by the EpwBaseWorkChain.

        inputs.metadata.call_link_label = "epw_base"

        workchain_node = self.submit(EpwBaseWorkChain, **inputs)
        self.report(
            f"launching EpwBaseWorkChain<{workchain_node.pk}> in transformation mode"
        )

        return ToContext(workchain_epw=workchain_node)

    def inspect_epw(self):
        """Verify that the `EpwBaseWorkChain` finished successfully."""
        workchain = self.ctx.workchain_epw

        if not workchain.is_finished_ok:
            self.report(format_subprocess_failure(workchain, "EpwBaseWorkChain"))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPW

    def should_run_epw_bands(self):
        """Check if the bands interpolation should be run."""
        return "epw_bands" in self.inputs

    def run_epw_bands(self):
        """Run the `EpwBaseWorkChain` in bands interpolation mode."""
        inputs = AttributeDict(
            self.exposed_inputs(EpwBaseWorkChain, namespace="epw_bands")
        )
        inputs.structure = self.inputs.structure
        inputs.parent_folder_epw = self.ctx.workchain_epw.outputs.remote_stash
        
        if 'workchain_w90_bands' in self.ctx:
            if "bands_kpoints" in self.ctx.workchain_w90_bands.inputs:
                bands_kpoints = self.ctx.workchain_w90_bands.inputs.bands_kpoints
            elif self.ctx.workchain_w90_bands:
                bands_kpoints = (
                    self.ctx.workchain_w90_bands.base.links.get_outgoing(
                        link_label_filter="seekpath_structure_analysis"
                )
                .first()
                .node.outputs.explicit_kpoints
            )
        else:
            from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
                seekpath_structure_analysis,
            )

            args = {
                "structure": self.inputs.structure,
                "metadata": {"call_link_label": "seekpath_structure_analysis"},
            }
            if "bands_kpoints_distance" in self.inputs:
                args["reference_distance"] = self.inputs["bands_kpoints_distance"]

            result = seekpath_structure_analysis(**args)
            bands_kpoints = result['explicit_kpoints']

        inputs.kpoints = self.ctx.kpoints_nscf
        inputs.qpoints = self.ctx.qpoints
        inputs.qfpoints = bands_kpoints
        inputs.kfpoints = bands_kpoints

        inputs.metadata.call_link_label = "epw_bands"
        workchain_node = self.submit(EpwBaseWorkChain, **inputs)
        self.report(
            f"launching EpwBaseWorkChain<{workchain_node.pk}> in bands interpolation mode"
        )

        return ToContext(workchain_epw_bands=workchain_node)

    def inspect_epw_bands(self):
        """Verify that the `EpwBaseWorkChain` in bands interpolation mode finished successfully."""
        workchain = self.ctx.workchain_epw_bands
        if not workchain.is_finished_ok:
            self.report(format_subprocess_failure(workchain, "EpwBaseWorkChain"))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EPW_BANDS

    def results(self):
        """Add the most important results to the outputs of the work chain."""
        self.out("retrieved", self.ctx.workchain_epw.outputs.retrieved)
        self.out("epw_folder", self.ctx.workchain_epw.outputs.remote_stash)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, exceptions.NotExistentAttributeError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
