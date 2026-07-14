"""Pure helpers for post-processing parsed EPW gap-function data."""

import numpy


def find_multigap_averages(data, temperature=None, bandwidth_factor=1.5):
    """Find representative gap peaks from a smoothed gap-distribution signal."""
    from scipy.signal import find_peaks

    data = numpy.array(data, dtype=float)
    gaps = data[:, 1]
    base_value = numpy.min(data[:, 0])
    signal = data[:, 0] - base_value

    max_sig = numpy.max(signal)
    if max_sig == 0:
        return [float(numpy.mean(gaps))]

    weights = signal / max_sig
    window = max(3, int(len(weights) * 0.03 * bandwidth_factor))
    if window % 2 == 0:
        window += 1
    kernel = numpy.ones(window) / window
    density = numpy.convolve(weights, kernel, mode="same")

    peaks, _ = find_peaks(density, prominence=numpy.max(density) * 0.1)

    if len(peaks) == 0:
        return [float(gaps[numpy.argmax(density)])]

    return sorted(float(gap) for gap in gaps[peaks])
