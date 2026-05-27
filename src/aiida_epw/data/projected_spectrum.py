"""Domain-specific data type for EPW projected spectra."""

import numpy
from aiida import orm
from aiida.common import exceptions


class ProjectedSpectrumData(orm.ArrayData):
    """Store a projected spectrum with explicit grid and split total/projected series."""

    ARRAY_GRID = "grid"
    ARRAY_SERIES = "series"
    ARRAY_TOTAL = "total"
    ARRAY_PROJECTED = "projected"

    ATTRIBUTE_KIND = "kind"
    ATTRIBUTE_GRID_NAME = "grid_name"
    ATTRIBUTE_SERIES_NAME = "series_name"
    ATTRIBUTE_TOTAL_LABEL = "total_label"
    ATTRIBUTE_PROJECTED_LABEL = "projected_label"
    ATTRIBUTE_LEGACY_GRID_NAME = "legacy_grid_name"
    ATTRIBUTE_LEGACY_SERIES_NAME = "legacy_series_name"

    def set_projected_spectrum(
        self,
        grid,
        series,
        *,
        kind,
        grid_name=None,
        series_name=None,
        total_label=None,
        projected_label=None,
        legacy_grid_name=None,
        legacy_series_name=None,
    ):
        """Store a projected spectrum and split off the total series."""
        grid = numpy.array(grid, dtype=float)
        series = numpy.array(series, dtype=float)

        if grid.ndim != 1:
            raise exceptions.ValidationError("`grid` must be a one-dimensional array.")
        if series.ndim != 2:
            raise exceptions.ValidationError(
                "`series` must be a two-dimensional array."
            )
        if series.shape[0] != grid.shape[0]:
            raise exceptions.ValidationError(
                "The first spectrum dimension must match the grid length."
            )
        if series.shape[1] < 1:
            raise exceptions.ValidationError(
                "The spectrum must contain at least one series column."
            )

        self._delete_alias_array(
            self.base.attributes.get(self.ATTRIBUTE_LEGACY_GRID_NAME, None)
        )
        self._delete_alias_array(
            self.base.attributes.get(self.ATTRIBUTE_LEGACY_SERIES_NAME, None)
        )

        self.set_array(self.ARRAY_GRID, grid)
        self.set_array(self.ARRAY_SERIES, series)
        self.set_array(self.ARRAY_TOTAL, series[:, 0])
        if series.shape[1] == 1:
            self._delete_alias_array(self.ARRAY_PROJECTED)
        else:
            self.set_array(self.ARRAY_PROJECTED, series[:, 1:])

        if legacy_grid_name is not None:
            self.set_array(legacy_grid_name, grid)
        if legacy_series_name is not None:
            self.set_array(legacy_series_name, series)

        self.base.attributes.set(self.ATTRIBUTE_KIND, kind)
        self._set_optional_attribute(self.ATTRIBUTE_GRID_NAME, grid_name)
        self._set_optional_attribute(self.ATTRIBUTE_SERIES_NAME, series_name)
        self._set_optional_attribute(self.ATTRIBUTE_TOTAL_LABEL, total_label)
        self._set_optional_attribute(self.ATTRIBUTE_PROJECTED_LABEL, projected_label)
        self._set_optional_attribute(self.ATTRIBUTE_LEGACY_GRID_NAME, legacy_grid_name)
        self._set_optional_attribute(
            self.ATTRIBUTE_LEGACY_SERIES_NAME, legacy_series_name
        )

    def get_grid(self):
        """Return the spectrum grid."""
        return self.get_array(self.ARRAY_GRID)

    def get_series(self):
        """Return the combined total-plus-projected spectrum matrix."""
        return self.get_array(self.ARRAY_SERIES)

    def get_total(self):
        """Return the total spectrum."""
        return self.get_array(self.ARRAY_TOTAL)

    def get_projected(self):
        """Return the projected spectrum columns, if present."""
        if self.ARRAY_PROJECTED not in self.get_arraynames():
            return None

        return self.get_array(self.ARRAY_PROJECTED)

    @property
    def kind(self):
        """Return the spectrum kind, e.g. `a2f_proj` or `phdos_proj`."""
        return self.base.attributes.get(self.ATTRIBUTE_KIND)

    @property
    def grid_name(self):
        """Return the semantic name of the grid."""
        return self.base.attributes.get(self.ATTRIBUTE_GRID_NAME, None)

    @property
    def series_name(self):
        """Return the semantic name of the combined series block."""
        return self.base.attributes.get(self.ATTRIBUTE_SERIES_NAME, None)

    @property
    def total_label(self):
        """Return the label of the total spectrum column."""
        return self.base.attributes.get(self.ATTRIBUTE_TOTAL_LABEL, None)

    @property
    def projected_label(self):
        """Return the label of the projected spectrum block."""
        return self.base.attributes.get(self.ATTRIBUTE_PROJECTED_LABEL, None)

    def _delete_alias_array(self, name):
        """Delete an alias array if it exists."""
        if name and name in self.get_arraynames():
            self.delete_array(name)

    def _set_optional_attribute(self, key, value):
        """Set or clear an optional scalar attribute."""
        if value is None:
            try:
                self.base.attributes.delete(key)
            except AttributeError:
                pass
            return

        self.base.attributes.set(key, value)

    @classmethod
    def from_string(cls, content, kind):
        """Instantiate and populate a `ProjectedSpectrumData` node from projected spectrum string content.

        :param content: string content to parse.
        :param kind: kind of projected spectrum ('a2f_proj' or 'phdos_proj').
        """
        from aiida_epw.tools.parsers import parse_epw_a2f_proj, parse_epw_phdos_proj

        if kind == "a2f_proj":
            parsed = parse_epw_a2f_proj(content)
            grid_name = "frequency"
            series_name = "a2f_proj"
            legacy_grid_name = "frequency"
            legacy_series_name = "a2f_proj"
        elif kind == "phdos_proj":
            parsed = parse_epw_phdos_proj(content)
            grid_name = "frequency"
            series_name = "phdos_proj"
            legacy_grid_name = "Frequency"
            legacy_series_name = "PHDOS_proj"
        else:
            raise ValueError(
                f"Unknown kind '{kind}': Must be 'a2f_proj' or 'phdos_proj'."
            )

        node = cls()
        node.set_projected_spectrum(
            grid=parsed["frequency"],
            series=parsed[series_name],
            kind=kind,
            grid_name=grid_name,
            series_name=series_name,
            total_label=parsed["total_label"],
            projected_label=parsed["projected_label"],
            legacy_grid_name=legacy_grid_name,
            legacy_series_name=legacy_series_name,
        )
        return node

    @classmethod
    def from_file(cls, filepath, kind):
        """Instantiate and populate a `ProjectedSpectrumData` node from a projected spectrum file."""
        from pathlib import Path

        content = Path(filepath).read_text(encoding="utf-8")
        return cls.from_string(content, kind=kind)

    @classmethod
    def from_a2f_proj(cls, content_or_filepath):
        """Instantiate and populate a `ProjectedSpectrumData` node of kind 'a2f_proj'."""
        from pathlib import Path

        try:
            if Path(content_or_filepath).is_file():
                return cls.from_file(content_or_filepath, kind="a2f_proj")
        except OSError:
            pass
        return cls.from_string(content_or_filepath, kind="a2f_proj")

    @classmethod
    def from_phdos_proj(cls, content_or_filepath):
        """Instantiate and populate a `ProjectedSpectrumData` node of kind 'phdos_proj'."""
        from pathlib import Path

        try:
            if Path(content_or_filepath).is_file():
                return cls.from_file(content_or_filepath, kind="phdos_proj")
        except OSError:
            pass
        return cls.from_string(content_or_filepath, kind="phdos_proj")
