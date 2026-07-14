"""Tests for plotting helpers using plain Python data structures."""

import matplotlib
import numpy

matplotlib.use("Agg")


def test_gap_iso_imag_temp_accepts_plain_dict(monkeypatch):
    """Test isotropic gap plotting accepts dictionaries from gap-data serialization."""
    from matplotlib import pyplot

    from aiida_epw.tools.plot import gap_iso_imag_temp

    monkeypatch.setattr(pyplot, "show", lambda: None)

    gap_iso_imag_temp(
        {
            "imag": {
                3.0: {
                    "columns": ["omega", "znorm", "deltaw"],
                    "table": numpy.array([[0.1, 1.0, 0.002], [0.2, 1.1, 0.001]]),
                    "data": {
                        "omega": numpy.array([0.1, 0.2]),
                        "znorm": numpy.array([1.0, 1.1]),
                        "deltaw": numpy.array([0.002, 0.001]),
                    },
                },
                4.0: {
                    "columns": ["omega", "znorm", "deltaw"],
                    "table": numpy.array([[0.1, 1.0, 0.001]]),
                    "data": {
                        "omega": numpy.array([0.1]),
                        "znorm": numpy.array([1.0]),
                        "deltaw": numpy.array([0.001]),
                    },
                },
            }
        },
        tempmax=5.0,
        fit=False,
    )


def test_gap_iso_imag_temp_accepts_gap_series(monkeypatch):
    """Test isotropic gap plotting accepts the compact Fermi-surface series."""
    from matplotlib import pyplot

    from aiida_epw.tools.plot import gap_iso_imag_temp

    monkeypatch.setattr(pyplot, "show", lambda: None)

    gap_iso_imag_temp(
        {"T": [3.0, 4.0], "gap": [2.0, 1.0], "unit": "meV", "source": "imag"},
        tempmax=5.0,
        fit=False,
    )


def test_plot_anisotropic_gap_accepts_plain_dict():
    """Test anisotropic gap plotting accepts dictionaries from gap-data serialization."""
    from matplotlib import pyplot

    from aiida_epw.tools.plot import plot_anisotropic_gap

    _, ax = pyplot.subplots()
    plot_anisotropic_gap(
        {
            "imag": {
                3.0: {
                    "columns": [
                        "T_dist_scaled",
                        "delta_nk",
                        "T",
                        "dist_scaled",
                        "dist_not_scaled",
                    ],
                    "table": numpy.array(
                        [
                            [3.0, 1.0, 3.0, 0.0, 0.0],
                            [3.5, 1.1, 3.0, 0.5, 5.0],
                            [3.1, 1.2, 3.0, 0.1, 1.0],
                        ]
                    ),
                }
            }
        },
        ax=ax,
        fit=False,
    )


def test_plot_anisotropic_gap_accepts_gap_series():
    """Test anisotropic gap plotting accepts precomputed representative gaps."""
    from matplotlib import pyplot

    from aiida_epw.tools.plot import plot_anisotropic_gap

    _, ax = pyplot.subplots()
    plot_anisotropic_gap(
        {"T": [3.0, 4.0, 5.0], "gap": [[1.2, 2.0], [1.0, 1.7], [0.8, 1.3]]},
        ax=ax,
        fit=False,
    )


def test_plot_anisotropic_frequency_gap_trims_low_frequencies():
    """Test frequency-resolved anisotropic plotting trims the low-frequency tail."""
    from matplotlib import pyplot

    from aiida_epw.tools.plot import plot_anisotropic_frequency_gap

    _, ax = pyplot.subplots()
    plot_anisotropic_frequency_gap(
        {
            3.0: {
                0.01: {"energy": [0.0], "delta": [-1.0e-5]},
                0.2: {"energy": [0.0], "delta": [2.0e-4]},
                "units": {"frequency": "eV", "energy": "eV", "delta": "eV"},
            }
        },
        ax=ax,
    )

    assert len(ax.lines) == 2
    plotted_gap_line = ax.lines[0]
    assert plotted_gap_line.get_label() == r"$\omega$=0.2 eV"


def test_plot_anisotropic_frequency_gap_accepts_explicit_frequencies():
    """Test explicit frequency selection bypasses the default low-frequency trim."""
    from matplotlib import pyplot

    from aiida_epw.tools.plot import plot_anisotropic_frequency_gap

    _, ax = pyplot.subplots()
    plot_anisotropic_frequency_gap(
        {
            3.0: {
                0.01: {"energy": [0.0], "delta": [-1.0e-5]},
                0.2: {"energy": [0.0], "delta": [2.0e-4]},
            }
        },
        frequencies=[0.01],
        ax=ax,
    )

    assert len(ax.lines) == 2
    plotted_gap_line = ax.lines[0]
    assert plotted_gap_line.get_label() == r"$\omega$=0.01 eV"
