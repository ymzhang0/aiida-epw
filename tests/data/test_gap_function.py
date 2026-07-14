"""Tests for the gap-function data types."""

import numpy
import pytest
from aiida.common import exceptions
from aiida.plugins import DataFactory

from aiida_epw.data import AnisoGap0Data, IsoGapData


def test_iso_gap_data_roundtrip_ragged_sources():
    """Test storing isotropic gap columns with ragged temperature grids."""
    node = IsoGapData()
    node.set_gap_data(
        {
            ("imag", 3.0): {
                "omega": [0.1, 0.2],
                "znorm": [1.0, 1.1],
                "deltaw": [0.01, 0.02],
            },
            ("pade", 3.0): {
                "omega": [0.1],
                "znorm_real": [1.0],
                "znorm_imag": [0.0],
                "deltaw_real": [0.01],
                "deltaw_imag": [0.001],
            },
        }
    )

    assert node.sources == ["imag", "pade"]
    assert sorted(node.get_arraynames()) == [
        "imag_003_00",
        "pade_003_00",
        "temperatures",
    ]
    assert node.get_temperatures(source="imag").tolist() == [3.0]
    assert node.get_table(3.0, source="imag").tolist() == [
        [0.1, 1.0, 0.01],
        [0.2, 1.1, 0.02],
    ]
    assert node.get_data(3.0, source="imag")["deltaw"].tolist() == [0.01, 0.02]
    assert node.get_data(3.0, source="pade")["deltaw_imag"].tolist() == [0.001]

    with pytest.raises(KeyError, match="Multiple gap data entries"):
        node.get_data(3.0)


def test_aniso_gap0_data_roundtrip_ragged_temperatures():
    """Test storing anisotropic gap0 distributions with different bin counts."""
    node = AnisoGap0Data()
    node.set_gap_data(
        {
            ("imag", 3.0): {
                "T_dist_scaled": [3.0, 3.1],
                "delta_nk": [1.0, 1.1],
                "T": [3.0, 3.0],
                "dist_scaled": [0.0, 0.1],
                "dist_not_scaled": [0.0, 10.0],
            },
            ("imag", 4.0): {
                "T_dist_scaled": [4.0],
                "delta_nk": [1.4],
                "T": [4.0],
                "dist_scaled": [0.2],
                "dist_not_scaled": [20.0],
            },
        }
    )

    assert node.get_temperatures(source="imag").tolist() == [3.0, 4.0]
    assert node.get_data(3.0, source="imag")["delta_nk"].shape == (2,)
    assert node.get_data(4.0, source="imag")["delta_nk"].shape == (1,)


def test_gap_data_replaces_previous_arrays():
    """Test resetting gap data drops stale arrays from a previous payload."""
    node = IsoGapData()
    node.set_gap_data(
        {
            ("imag", 3.0): {
                "omega": [0.1, 0.2],
                "znorm": [1.0, 1.1],
                "deltaw": [0.01, 0.02],
            }
        }
    )
    node.set_gap_data(
        {
            ("imag", 4.0): {
                "omega": [0.3],
                "znorm": [1.3],
                "deltaw": [0.03],
            }
        }
    )

    assert sorted(node.get_arraynames()) == [
        "imag_004_00",
        "temperatures",
    ]


def test_gap_data_validates_shape_contract():
    """Test invalid gap-function payloads are rejected."""
    node = IsoGapData()

    with pytest.raises(exceptions.ValidationError):
        node.set_gap_data({})

    with pytest.raises(exceptions.ValidationError):
        node.set_gap_data(
            {
                ("imag", 3.0): {
                    "omega": numpy.array([[0.1, 0.2]]),
                    "znorm": [1.0, 1.1],
                    "deltaw": [0.01, 0.02],
                }
            }
        )

    with pytest.raises(exceptions.ValidationError):
        node.set_gap_data(
            {
                ("imag", 3.0): {
                    "omega": [0.1, 0.2],
                    "znorm": [1.0],
                    "deltaw": [0.01, 0.02],
                }
            }
        )


def test_gap_data_entry_points():
    """Test the datatypes are registered through ``aiida.data``."""
    assert DataFactory("epw.iso_gap") is IsoGapData
    assert DataFactory("epw.aniso_gap0") is AnisoGap0Data


def test_gap_data_serialization_factories(files_path):
    """Test from_files and from_directory classmethods."""
    iso_dir = files_path / "tools" / "parsers" / "full_iso_eliashberg"
    iso_node = IsoGapData.from_directory(iso_dir, prefix="aiida")

    assert isinstance(iso_node, IsoGapData)
    assert iso_node.sources == ["imag"]
    assert iso_node.get_temperatures(source="imag").tolist() == [3.0, 4.0, 5.0]
    assert set(iso_node.get_data(3.0, source="imag")) == {"omega", "znorm", "deltaw"}

    aniso_dir = files_path / "tools" / "parsers" / "fsr_aniso_eliashberg"
    aniso_node = AnisoGap0Data.from_directory(aniso_dir, prefix="aiida")

    assert isinstance(aniso_node, AnisoGap0Data)
    assert aniso_node.sources == ["imag"]
    assert aniso_node.get_temperatures(source="imag").tolist() == [3.0, 4.0, 5.0]
    assert set(aniso_node.get_data(3.0, source="imag")) == {
        "T_dist_scaled",
        "delta_nk",
        "T",
        "dist_scaled",
        "dist_not_scaled",
    }
