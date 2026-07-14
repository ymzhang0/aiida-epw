"""Domain-specific data types for EPW gap-function output tables."""

from pathlib import Path

import numpy
from aiida import orm
from aiida.common import exceptions


def _temperature_label(temperature):
    """Return a stable array-name fragment for a temperature value."""
    return f"{float(temperature):06.2f}".replace(".", "_")


class _RaggedGapData(orm.ArrayData):
    """Store gap output tables without assuming equal row counts across temperatures."""

    ATTRIBUTE_ENTRIES = "entries"
    ARRAY_TEMPERATURES = "temperatures"

    def set_gap_data(self, gap_data):
        """Store a mapping keyed by `(source, temperature)` as one table per entry."""
        if not gap_data:
            raise exceptions.ValidationError("`gap_data` cannot be empty.")

        self._delete_gap_arrays()
        entries = []
        temperatures = []

        for (source, temperature), columns in sorted(
            gap_data.items(), key=lambda item: (item[0][0], float(item[0][1]))
        ):
            if not columns:
                raise exceptions.ValidationError(
                    "Gap-data column mappings cannot be empty."
                )

            source = str(source)
            temperature = float(temperature)
            label = f"{source}_{_temperature_label(temperature)}"
            row_count = None
            column_names = []
            column_arrays = []

            for column_name, values in columns.items():
                array = numpy.array(values, dtype=float)
                if array.ndim != 1:
                    raise exceptions.ValidationError(
                        f"Column `{column_name}` for {label} must be one-dimensional."
                    )
                if row_count is None:
                    row_count = array.shape[0]
                elif array.shape[0] != row_count:
                    raise exceptions.ValidationError(
                        f"Columns for {label} must have the same length."
                    )

                column_names.append(column_name)
                column_arrays.append(array)

            self.set_array(label, numpy.column_stack(column_arrays))

            entries.append(
                {
                    "source": source,
                    "temperature": temperature,
                    "label": label,
                    "array_name": label,
                    "columns": column_names,
                }
            )
            temperatures.append(temperature)

        self.set_array(self.ARRAY_TEMPERATURES, numpy.array(temperatures, dtype=float))
        self.base.attributes.set(self.ATTRIBUTE_ENTRIES, entries)

    def get_temperatures(self, source=None):
        """Return the stored temperatures in Kelvin, optionally filtered by source."""
        temperatures = [
            entry["temperature"] for entry in self._get_entries(source=source)
        ]
        return numpy.array(temperatures, dtype=float)

    def get_data(self, temperature, *, source=None, atol=1e-8):
        """Return named arrays for a specific temperature and optional source."""
        entry = self._find_entry(temperature, source=source, atol=atol)
        table = self.get_array(entry["array_name"])
        return {
            column: table[:, index] for index, column in enumerate(entry["columns"])
        }

    def get_table(self, temperature, *, source=None, atol=1e-8):
        """Return the stored two-dimensional table for a temperature/source entry."""
        entry = self._find_entry(temperature, source=source, atol=atol)
        return self.get_array(entry["array_name"])

    def get_iterdata(self, source=None):
        """Yield `(source, temperature, columns)` in stored order."""
        for entry in self._get_entries(source=source):
            yield (
                entry["source"],
                entry["temperature"],
                self.get_data(entry["temperature"], source=entry["source"]),
            )

    def to_dict(self, source=None):
        """Return the stored gap data as plain dictionaries and NumPy arrays."""
        data = {}
        for entry in self._get_entries(source=source):
            table = self.get_array(entry["array_name"])
            columns = list(entry["columns"])
            data.setdefault(entry["source"], {})[entry["temperature"]] = {
                "columns": columns,
                "table": table,
                "data": {
                    column: table[:, index] for index, column in enumerate(columns)
                },
            }
        return data

    @property
    def sources(self):
        """Return the source labels represented by this node, e.g. `imag` or `pade`."""
        return sorted({entry["source"] for entry in self._get_entries()})

    def _get_entries(self, source=None):
        """Return metadata entries, optionally filtered by source."""
        entries = self.base.attributes.get(self.ATTRIBUTE_ENTRIES, [])
        if source is not None:
            entries = [entry for entry in entries if entry["source"] == source]
        return entries

    def _find_entry(self, temperature, *, source=None, atol=1e-8):
        """Find the metadata entry for a temperature/source pair."""
        target = float(temperature)
        matches = [
            entry
            for entry in self._get_entries(source=source)
            if numpy.isclose(entry["temperature"], target, atol=atol, rtol=0.0)
        ]

        if not matches:
            suffix = f" and source `{source}`" if source is not None else ""
            raise KeyError(f"No gap data stored for temperature {target}{suffix}.")
        if len(matches) > 1:
            raise KeyError(
                f"Multiple gap data entries found for temperature {target}; pass `source`."
            )
        return matches[0]

    def _delete_gap_arrays(self):
        """Delete arrays referenced by previous gap-data entries."""
        for entry in self.base.attributes.get(self.ATTRIBUTE_ENTRIES, []):
            array_name = entry.get("array_name", entry.get("label"))
            if array_name in self.get_arraynames():
                self.delete_array(array_name)
        if self.ARRAY_TEMPERATURES in self.get_arraynames():
            self.delete_array(self.ARRAY_TEMPERATURES)


class IsoGapData(_RaggedGapData):
    """Store isotropic EPW gap-function columns by source and temperature."""

    def get_gap_FS(self, source="imag", component=None, unit="meV", drop_nan=True):
        """Return the Fermi-surface gap as a plain temperature series."""
        factor = {"eV": 1.0, "meV": 1000.0}[unit]
        temperatures = []
        gaps = []

        for _, temperature, columns in self.get_iterdata(source=source):
            if component is None:
                if "deltaw" in columns:
                    column_name = "deltaw"
                elif "deltaw_real" in columns:
                    column_name = "deltaw_real"
                else:
                    raise KeyError(
                        "Could not find a gap column; pass `component` explicitly."
                    )
            else:
                column_name = component

            gap = float(columns[column_name][0]) * factor
            if drop_nan and numpy.isnan(gap):
                continue

            temperatures.append(float(temperature))
            gaps.append(gap)

        return {"T": temperatures, "gap": gaps, "unit": unit, "source": source}

    def get_gap_fs(self, *args, **kwargs):
        """Alias for :meth:`get_gap_FS` using conventional Python casing."""
        return self.get_gap_FS(*args, **kwargs)

    @classmethod
    def from_files(cls, file_contents_or_paths, prefix="aiida"):
        """Instantiate and populate an `IsoGapData` node from isotropic gap files."""
        from aiida_epw.tools.parsers import parse_epw_iso_gap_files

        node = cls()
        node.set_gap_data(
            parse_epw_iso_gap_files(file_contents_or_paths, prefix=prefix)
        )
        return node

    @classmethod
    def from_directory(cls, dirpath, prefix="aiida"):
        """Instantiate and populate an `IsoGapData` node from a directory."""
        path = Path(dirpath)
        filepaths = [
            filepath
            for pattern in (f"{prefix}.imag_iso_*", f"{prefix}.pade_iso_*")
            for filepath in path.glob(pattern)
        ]
        if not filepaths:
            raise FileNotFoundError(
                f"No files matching '{prefix}.imag_iso_*' or '{prefix}.pade_iso_*' "
                f"in directory '{dirpath}'"
            )

        return cls.from_files(filepaths, prefix=prefix)


class AnisoGap0Data(_RaggedGapData):
    """Store anisotropic gap0 distribution columns by source and temperature."""

    def get_multigap_averages(self, source="imag", bandwidth_factor=1.5):
        """Return representative anisotropic gap values for each temperature."""
        from aiida_epw.tools.gap import find_multigap_averages

        temperatures = []
        gaps = []

        for _, temperature, _ in self.get_iterdata(source=source):
            table = self.get_table(temperature, source=source)
            temperatures.append(float(temperature))
            gaps.append(
                find_multigap_averages(
                    table,
                    temperature=temperature,
                    bandwidth_factor=bandwidth_factor,
                )
            )

        return {"T": temperatures, "gap": gaps, "source": source}

    @classmethod
    def from_files(cls, file_contents_or_paths, prefix="aiida"):
        """Instantiate and populate an `AnisoGap0Data` node from aniso gap0 files."""
        from aiida_epw.tools.parsers import parse_epw_aniso_gap0_files

        node = cls()
        node.set_gap_data(
            parse_epw_aniso_gap0_files(file_contents_or_paths, prefix=prefix)
        )
        return node

    @classmethod
    def from_directory(cls, dirpath, prefix="aiida"):
        """Instantiate and populate an `AnisoGap0Data` node from a directory."""
        path = Path(dirpath)
        filepaths = [
            filepath
            for pattern in (
                f"{prefix}.imag_aniso_gap0_*",
                f"{prefix}.pade_aniso_gap0_*",
            )
            for filepath in path.glob(pattern)
        ]
        if not filepaths:
            raise FileNotFoundError(
                f"No files matching '{prefix}.imag_aniso_gap0_*' or "
                f"'{prefix}.pade_aniso_gap0_*' in directory '{dirpath}'"
            )

        return cls.from_files(filepaths, prefix=prefix)
