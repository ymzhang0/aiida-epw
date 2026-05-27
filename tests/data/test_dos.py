"""Tests for the ``DosData`` datatype."""

import pytest
from aiida.common import exceptions
from aiida.plugins import DataFactory

from aiida_epw.data import DosData


def test_dos_data_roundtrip():
    """Test storing and retrieving a DOS dataset."""
    node = DosData()
    node.set_dos_data(
        energy=[10.0, 11.0, 12.0],
        edos=[0.5, 1.2, 0.8],
        integrated_dos=[0.05, 0.17, 0.25],
    )

    assert node.get_energy().tolist() == [10.0, 11.0, 12.0]
    assert node.get_edos().tolist() == [0.5, 1.2, 0.8]
    assert node.get_integrated_dos().tolist() == [0.05, 0.17, 0.25]

    # Test legacy aliases for compatibility with XyData usages
    assert node.get_array("Energy").tolist() == [10.0, 11.0, 12.0]
    assert node.get_array("EDOS").tolist() == [0.5, 1.2, 0.8]
    assert node.get_array("IDOS").tolist() == [0.05, 0.17, 0.25]


def test_dos_data_optional_integrated_dos():
    """Test storing and retrieving a DOS dataset without integrated DOS."""
    node = DosData()
    node.set_dos_data(
        energy=[10.0, 11.0, 12.0],
        edos=[0.5, 1.2, 0.8],
    )

    assert node.get_energy().tolist() == [10.0, 11.0, 12.0]
    assert node.get_edos().tolist() == [0.5, 1.2, 0.8]
    assert node.get_integrated_dos() is None

    # Verify IDOS KeyError behaves compatibly with get_integrated_dos helper
    with pytest.raises(KeyError):
        node.get_array("IDOS")


def test_dos_data_validates_shape_contract():
    """Test that invalid DOS payloads are rejected."""
    node = DosData()

    # Mismatched lengths
    with pytest.raises(exceptions.ValidationError):
        node.set_dos_data(
            energy=[10.0, 11.0],
            edos=[0.5, 1.2, 0.8],
        )

    # Inconsistent integrated_dos length
    with pytest.raises(exceptions.ValidationError):
        node.set_dos_data(
            energy=[10.0, 11.0, 12.0],
            edos=[0.5, 1.2, 0.8],
            integrated_dos=[0.05, 0.17],
        )

    # Multi-dimensional arrays
    with pytest.raises(exceptions.ValidationError):
        node.set_dos_data(
            energy=[[10.0]],
            edos=[0.5],
        )


def test_dos_data_entry_point():
    """Test the datatype is registered through ``aiida.data``."""
    assert DataFactory("epw.dos") is DosData


def test_dos_serialization_factories(files_path):
    """Test DosData.from_file and from_string classmethods."""
    dos_file = files_path / "tools" / "parsers" / "a2f" / "aiida.dos"

    # Test from_file
    node_file = DosData.from_file(dos_file)
    assert isinstance(node_file, DosData)
    assert node_file.get_energy().shape == (160,)
    assert node_file.get_edos().shape == (160,)
    assert node_file.get_integrated_dos().shape == (160,)
    assert node_file.get_energy()[0] == pytest.approx(10.871190406)

    # Test from_string
    content = dos_file.read_text(encoding="utf-8")
    node_str = DosData.from_string(content)
    assert isinstance(node_str, DosData)
    assert node_str.get_energy().tolist() == node_file.get_energy().tolist()
