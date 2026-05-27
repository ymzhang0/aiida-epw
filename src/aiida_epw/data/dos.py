"""Domain-specific data type for EPW electronic Density of States (DOS)."""

import numpy
from aiida import orm
from aiida.common import exceptions


class DosData(orm.ArrayData):
    """Store the EPW electronic DOS table with explicit semantic getters."""

    ARRAY_ENERGY = "energy"
    ARRAY_EDOS = "edos"
    ARRAY_INTEGRATED_DOS = "integrated_dos"

    # Legacy array keys for compatibility with generic XyData usages
    LEGACY_ARRAY_ENERGY = "Energy"
    LEGACY_ARRAY_EDOS = "EDOS"
    LEGACY_ARRAY_INTEGRATED_DOS = "IDOS"

    def set_dos_data(self, energy, edos, integrated_dos=None):
        """Store the electronic DOS arrays."""
        energy = numpy.array(energy, dtype=float)
        edos = numpy.array(edos, dtype=float)

        if energy.ndim != 1:
            raise exceptions.ValidationError(
                "`energy` must be a one-dimensional array."
            )
        if edos.ndim != 1:
            raise exceptions.ValidationError("`edos` must be a one-dimensional array.")
        if energy.shape[0] != edos.shape[0]:
            raise exceptions.ValidationError(
                "`energy` and `edos` must have the same length."
            )

        self.set_array(self.ARRAY_ENERGY, energy)
        self.set_array(self.LEGACY_ARRAY_ENERGY, energy)
        self.set_array(self.ARRAY_EDOS, edos)
        self.set_array(self.LEGACY_ARRAY_EDOS, edos)

        if integrated_dos is not None:
            integrated_dos = numpy.array(integrated_dos, dtype=float)
            if integrated_dos.ndim != 1:
                raise exceptions.ValidationError(
                    "`integrated_dos` must be a one-dimensional array."
                )
            if integrated_dos.shape[0] != energy.shape[0]:
                raise exceptions.ValidationError(
                    "`integrated_dos` must have the same length as `energy`."
                )
            self.set_array(self.ARRAY_INTEGRATED_DOS, integrated_dos)
            self.set_array(self.LEGACY_ARRAY_INTEGRATED_DOS, integrated_dos)

    def get_energy(self):
        """Return the energy array."""
        return self.get_array(self.ARRAY_ENERGY)

    def get_edos(self):
        """Return the electronic DOS array."""
        return self.get_array(self.ARRAY_EDOS)

    def get_integrated_dos(self):
        """Return the integrated electronic DOS array, or None if not set."""
        try:
            return self.get_array(self.ARRAY_INTEGRATED_DOS)
        except KeyError:
            return None

    @classmethod
    def from_string(cls, content):
        """Instantiate and populate a `DosData` node directly from `.dos` string content."""
        from aiida_epw.tools.parsers import parse_epw_eldos

        parsed = parse_epw_eldos(content)
        node = cls()
        node.set_dos_data(
            energy=parsed["energy"],
            edos=parsed["edos"],
            integrated_dos=parsed.get("integrated_dos"),
        )
        return node

    @classmethod
    def from_file(cls, filepath):
        """Instantiate and populate a `DosData` node directly from a `.dos` file."""
        from pathlib import Path

        content = Path(filepath).read_text(encoding="utf-8")
        return cls.from_string(content)
