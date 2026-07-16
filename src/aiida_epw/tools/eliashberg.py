"""Helpers for adaptive Eliashberg temperature sampling."""

from __future__ import annotations

import math

import numpy


def get_temperature_list(parameters):
    """Return an explicit temperature list from EPW ``INPUTEPW`` parameters."""
    input_epw = parameters.get("INPUTEPW", {})
    temps = input_epw.get("temps")
    if temps is None:
        return []

    if isinstance(temps, str):
        values = [float(value) for value in temps.replace(",", " ").split()]
    elif isinstance(temps, (int, float)):
        values = [float(temps)]
    else:
        values = [float(value) for value in temps]

    nstemp = input_epw.get("nstemp")
    if len(values) == 2 and nstemp is not None and int(nstemp) > 2:
        return numpy.linspace(values[0], values[1], int(nstemp)).tolist()

    return values


def set_temperature_list(parameters, temperatures):
    """Return a copy of parameters with explicit ``temps`` and no ``nstemp``."""
    updated = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in parameters.items()
    }
    input_epw = updated.setdefault("INPUTEPW", {})
    input_epw["temps"] = sorted(float(temperature) for temperature in temperatures)
    input_epw.pop("nstemp", None)
    input_epw.pop("tempsmin", None)
    input_epw.pop("tempsmax", None)
    return updated


def representative_gap_series(gap_series):
    """Return sorted ``(temperature, representative_gap)`` pairs from a gap series."""
    pairs = []
    for temperature, gap_value in zip(
        gap_series.get("T", []), gap_series.get("gap", [])
    ):
        if isinstance(gap_value, (list, tuple, numpy.ndarray)):
            values = numpy.array(gap_value, dtype=float)
            finite_positive = values[numpy.isfinite(values) & (values > 0.0)]
            if finite_positive.size == 0:
                gap = math.nan
            else:
                gap = float(numpy.max(finite_positive))
        else:
            gap = float(gap_value)
        pairs.append((float(temperature), gap))

    return sorted(pairs)


def suggest_temperatures(
    temperatures,
    gap_series,
    *,
    gap_close_fraction=0.05,
    gap_open_fraction=0.2,
    gap_abs_tol=1.0e-3,
    min_valid_points=3,
    expand_factor=2.0,
    refine_points=5,
    max_temperature_points=30,
    min_temperature=0.1,
):
    """Analyze gap data and suggest an explicit next temperature list."""
    current_temperatures = sorted(float(temperature) for temperature in temperatures)
    pairs = representative_gap_series(gap_series)
    valid_pairs = [
        (temperature, gap) for temperature, gap in pairs if numpy.isfinite(gap)
    ]

    if len(valid_pairs) < min_valid_points:
        return {
            "status": "BAD_DATA",
            "reason": "Not enough finite gap values were parsed.",
            "temperatures": current_temperatures,
        }

    positive_gaps = [gap for _, gap in valid_pairs if gap > gap_abs_tol]
    if not positive_gaps:
        return {
            "status": "RANGE_TOO_HIGH",
            "reason": "All parsed gaps are below the absolute gap threshold.",
            "temperatures": _expand_lower(
                current_temperatures,
                expand_factor,
                min_temperature,
                max_temperature_points,
            ),
        }

    max_gap = max(positive_gaps)
    normalized = [(temperature, gap / max_gap) for temperature, gap in valid_pairs]
    lowest_temperature, lowest_gap = normalized[0]
    highest_temperature, highest_gap = normalized[-1]

    if lowest_gap < gap_open_fraction:
        return {
            "status": "RANGE_TOO_HIGH",
            "reason": (
                f"The lowest-temperature gap at {lowest_temperature:g} K is already "
                "too close to the closed-gap regime."
            ),
            "temperatures": _expand_lower(
                current_temperatures,
                expand_factor,
                min_temperature,
                max_temperature_points,
            ),
        }

    if highest_gap > gap_close_fraction:
        return {
            "status": "RANGE_TOO_LOW",
            "reason": (
                f"The highest-temperature gap at {highest_temperature:g} K is still "
                "above the closed-gap threshold."
            ),
            "temperatures": _expand_higher(
                current_temperatures, expand_factor, max_temperature_points
            ),
        }

    open_points = [
        temperature for temperature, gap in normalized if gap >= gap_open_fraction
    ]
    closed_points = [
        temperature for temperature, gap in normalized if gap <= gap_close_fraction
    ]
    if open_points and closed_points:
        lower = max(open_points)
        higher_candidates = [
            temperature for temperature in closed_points if temperature > lower
        ]
        if higher_candidates:
            higher = min(higher_candidates)
            bracket_count = sum(
                lower <= temperature <= higher for temperature in current_temperatures
            )
            if bracket_count < refine_points:
                return {
                    "status": "NEED_MORE_POINTS",
                    "reason": "The gap-closing region is bracketed but undersampled.",
                    "temperatures": _refine_between(
                        current_temperatures,
                        lower,
                        higher,
                        refine_points,
                        max_temperature_points,
                    ),
                }

    return {
        "status": "GOOD",
        "reason": "The sampled temperatures include open and closed gap regimes.",
        "temperatures": current_temperatures,
    }


def _expand_lower(temperatures, expand_factor, min_temperature, max_temperature_points):
    """Add lower-temperature points."""
    if not temperatures:
        return []
    lower = temperatures[0]
    additions = [
        max(min_temperature, lower / expand_factor**2),
        max(min_temperature, lower / expand_factor),
    ]
    return _merge_temperature_lists(additions + temperatures, max_temperature_points)


def _expand_higher(temperatures, expand_factor, max_temperature_points):
    """Add higher-temperature points."""
    if not temperatures:
        return []
    upper = temperatures[-1]
    additions = [upper * math.sqrt(expand_factor), upper * expand_factor]
    return _merge_temperature_lists(temperatures + additions, max_temperature_points)


def _refine_between(temperatures, lower, upper, refine_points, max_temperature_points):
    """Add evenly spaced points between a bracketing pair."""
    additions = numpy.linspace(lower, upper, refine_points).tolist()
    return _merge_temperature_lists(temperatures + additions, max_temperature_points)


def _merge_temperature_lists(temperatures, max_temperature_points):
    """Return sorted unique temperatures capped to a maximum length."""
    merged = sorted({round(float(temperature), 8) for temperature in temperatures})
    if len(merged) <= max_temperature_points:
        return merged

    indices = numpy.linspace(0, len(merged) - 1, max_temperature_points)
    return [merged[int(round(index))] for index in indices]
