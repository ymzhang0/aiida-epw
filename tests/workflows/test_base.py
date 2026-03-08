from aiida import orm

from aiida_epw.workflows.base import EpwBaseWorkChain


def generate_structure():
    """Return a minimal silicon structure for protocol builder tests."""
    structure = orm.StructureData(
        cell=[
            [0.0, 2.715, 2.715],
            [2.715, 0.0, 2.715],
            [2.715, 2.715, 0.0],
        ]
    )
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si")
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si")
    return structure


def test_get_builder_from_protocol_accepts_parallelization_override(fixture_code):
    """Test that protocol builders can forward the `parallelization` input."""
    builder = EpwBaseWorkChain.get_builder_from_protocol(
        code=fixture_code("epw.epw"),
        structure=generate_structure(),
        overrides={"parallelization": {"npool": 2}},
    )

    assert builder.parallelization.get_dict() == {"npool": 2}
    assert "parent_folder" not in builder
