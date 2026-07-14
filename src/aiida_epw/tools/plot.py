"""Plotting functions copied and adapted from EPWpy."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as numpy
from scipy.optimize import curve_fit

from aiida_epw.tools.calculators import bcs_gap_function


def plot_max_eigenvalue(temps, evs, ax=None, **kwargs):
    """Plot the isotropic gap (Imaginary) vs. temeprature."""
    import numpy

    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)
    title = kwargs.pop(
        "title",
        "Max. eigenvalue of linearized Eliashberg equation",
    )
    xlabel = kwargs.pop("xlabel", r"Temeperature (K)")
    ylabel = kwargs.pop("ylabel", r"Max. eigenvalue")

    ##Plot
    if not ax:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": kwargs.pop("fontsize", 12)})
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["STIXGeneral"]
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"
        plt.rcParams["mathtext.default"] = "regular"
        fig, axs = plt.subplots(
            1, 1, figsize=(5, 4), squeeze=False, constrained_layout=True
        )
        ax = axs[0, 0]

    ax.set_title(title)
    ax.set_xlim([numpy.min(temps), numpy.max(temps)] if not xlim else xlim)
    ax.set_ylim([numpy.min(evs), numpy.max(evs)] if not ylim else ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="y")
    ax.tick_params(axis="x")
    ax.plot(
        temps,
        evs,
        linestyle="-",
        marker="o",
        c="k",
    )

    ax.axhline(1.0, ls="--", c="gray", lw=1.5)
    # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def _source_mapping(gap_functions, source):
    """Return a temperature mapping from plain gap dictionaries."""
    if source in gap_functions and isinstance(gap_functions[source], dict):
        return gap_functions[source]
    return gap_functions


def _iter_iso_gap_data(gap_functions, source="imag"):
    """Yield `(temperature, columns)` from plain isotropic gap dictionaries."""
    if "T" in gap_functions and "gap" in gap_functions:
        for temperature, gap in zip(gap_functions["T"], gap_functions["gap"]):
            yield float(temperature), {"gap": numpy.array([gap], dtype=float)}
        return

    for temperature, entry in sorted(_source_mapping(gap_functions, source).items()):
        if isinstance(entry, dict) and "data" in entry:
            yield float(temperature), entry["data"]
        elif isinstance(entry, dict):
            yield float(temperature), entry
        else:
            table = numpy.array(entry, dtype=float)
            columns = {
                "omega": table[:, 0],
                "znorm": table[:, 1],
                "deltaw": table[:, 2],
            }
            if table.shape[1] > 3:
                columns["shift"] = table[:, 3]
            yield float(temperature), columns


def _iter_aniso_gap_tables(gap_functions, source="imag"):
    """Yield `(temperature, table)` from plain anisotropic gap dictionaries."""
    for temperature, entry in sorted(_source_mapping(gap_functions, source).items()):
        if isinstance(entry, dict) and "table" in entry:
            table = numpy.array(entry["table"], dtype=float)
        elif isinstance(entry, dict) and "data" in entry:
            data = entry["data"]
            table = numpy.column_stack(
                [
                    data["T_dist_scaled"],
                    data["delta_nk"],
                    data["T"],
                    data["dist_scaled"],
                    data["dist_not_scaled"],
                ]
            )
        elif isinstance(entry, dict):
            table = numpy.column_stack(
                [
                    entry["T_dist_scaled"],
                    entry["delta_nk"],
                    entry["T"],
                    entry["dist_scaled"],
                    entry["dist_not_scaled"],
                ]
            )
        else:
            table = numpy.array(entry, dtype=float)
        yield float(temperature), table


#### Isotropic gap (Imaginary, real and ) vs. temeprature


def gap_iso_imag_temp(
    iso_gap_function,
    tempmax,
    font=12,
    prefix="aiida",
    source="imag",
    fit=False,
    p0=None,
    destpath=None,
):
    """Plot the isotropic gap (Imaginary) vs. temeprature."""
    imag_delta = []
    imag_temp = []

    for temperature, columns in _iter_iso_gap_data(iso_gap_function, source=source):
        gap = columns["gap"][0] if "gap" in columns else columns["deltaw"][0] * 1000
        if numpy.isnan(gap):
            continue
        imag_delta.append(gap)  # Convert to meV
        imag_temp.append(temperature)

    ##Plot
    fig = plt.figure(figsize=(4.5, 3.5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("Superconducting Gap vs. Temperature", fontsize=font)
    ax1.set_xlabel("Temeperature (K)", fontsize=font)
    ax1.set_xlim(0, tempmax)
    ax1.set_ylabel(r"$\Delta_0$ (meV)", fontsize=font)
    ax1.tick_params(axis="y", labelsize=font)
    ax1.tick_params(axis="x", labelsize=font)
    ax1.plot(
        imag_temp,
        imag_delta,
        linestyle="-",
        marker="o",
        c="k",
        label="Im. axis",
    )
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    plt.tight_layout()
    if fit:
        if p0 is None:
            p0 = [imag_temp[-1], 3.3, imag_delta[0]]
        popt, pcov = curve_fit(bcs_gap_function, imag_temp, imag_delta, p0=p0)
        Tc, p, Delta_0 = popt
        T = numpy.linspace(0, Tc, 100)
        ax1.plot(
            T,
            bcs_gap_function(T, Tc, p, Delta_0),
            linestyle="--",
            c="r",
            label="Fit",
        )
    if destpath:
        plt.savefig(os.path.join(destpath, f"{prefix}_iso_gap_imag_vs_Temp.pdf"))
    plt.show()


def fitting_function(T, p, delta_zero, Tc):
    """
    Standard BCS-like fitting function for superconducting gap.
    Handles T > Tc safely by returning 0.
    """
    T = numpy.atleast_1d(T)
    gap = numpy.zeros_like(T, dtype=float)
    mask = T < Tc
    gap[mask] = delta_zero * (1.0 - (T[mask] / Tc) ** p) ** 0.5
    return gap if len(gap) > 1 else gap[0]


def plot_anisotropic_gap(
    aniso_gap_functions_dict,
    ax=None,
    source="imag",
    fit=True,
    p0=None,
    destpath=None,
    **kwargs,
):
    """Plot the anisotropic gap vs. temperature and fit multi-gap functions automatically."""
    from scipy.optimize import curve_fit

    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)
    title = kwargs.pop("title", "Multi-gap Fitting Analysis")
    xlabel = kwargs.pop("xlabel", r"Temperature (K)")
    ylabel = kwargs.pop("ylabel", r"$\Delta_{nk}$ (meV)")

    if not ax:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": kwargs.pop("fontsize", 12)})
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["STIXGeneral"]
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"
        plt.rcParams["mathtext.default"] = "regular"
        fig, axs = plt.subplots(
            1, 1, figsize=(6, 5), squeeze=False, constrained_layout=True
        )
        ax = axs[0, 0]

    # 用字典动态追踪不同的能隙分支：{branch_index: (list_of_T, list_of_delta)}
    branches = {}

    if "T" in aniso_gap_functions_dict and "gap" in aniso_gap_functions_dict:
        for T, rep_gaps in zip(
            aniso_gap_functions_dict["T"], aniso_gap_functions_dict["gap"]
        ):
            if numpy.isscalar(rep_gaps):
                rep_gaps = [rep_gaps]
            for idx, vg in enumerate(rep_gaps):
                branches.setdefault(idx, ([], []))
                branches[idx][0].append(T)
                branches[idx][1].append(vg)

            if rep_gaps:
                ax.scatter(
                    [T] * len(rep_gaps),
                    rep_gaps,
                    color="red",
                    edgecolors="black",
                    s=25,
                    zorder=5,
                )
    else:
        gap_tables = dict(
            _iter_aniso_gap_tables(aniso_gap_functions_dict, source=source)
        )
        sorted_temps = sorted(gap_tables.keys())

        if len(sorted_temps) > 1:
            dT = numpy.mean(numpy.diff(sorted_temps))
        else:
            dT = 3.0

        for T in sorted_temps:
            array = gap_tables[T]

            base_value = numpy.min(array[:, 0])
            signal = array[:, 0] - base_value
            max_sig = numpy.max(signal)

            if max_sig > 0:
                scale = (dT * 0.45) / max_sig
                ax.fill_betweenx(
                    y=array[:, 1],
                    x1=T - signal * scale,
                    x2=T + signal * scale,
                    color="tab:blue",
                    alpha=0.15,
                    edgecolor="tab:blue",
                    linewidth=0.5,
                    zorder=1,
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # 4. 动态对所有识别出的分支独立进行 BCS 拟合
    if fit:
        # 使用不同的颜色区分拟合曲线
        color_cycle = ["r", "g", "b", "m", "c"]

        for b_idx, (ts, ds) in branches.items():
            ts = numpy.array(ts)
            ds = numpy.array(ds)

            # 如果某个分支的数据点太少（例如高温下某些小能隙闭合了），则跳过拟合
            if len(ts) < 3:
                continue

            # 为当前分支做初始猜测 [p, delta_zero, Tc]
            if p0 is None:
                current_p0 = [3.0, ds[0], ts[-1] * 1.05]
            else:
                current_p0 = p0

            try:
                # 施加合理的物理边界：p在0.5~10之间，Delta_0大于0，Tc大于当前观测最高温度
                popt, pcov = curve_fit(
                    fitting_function,
                    ts,
                    ds,
                    p0=current_p0,
                    maxfev=10000,
                    bounds=((0.5, 0.0, ts[-1]), (10.0, ds[0] * 2, ts[-1] * 2)),
                )
                p_fit, delta_zero_fit, Tc_fit = popt

                # 绘制拟合线
                T_fit = numpy.linspace(0, Tc_fit, 100)
                ax.plot(
                    T_fit,
                    fitting_function(T_fit, p_fit, delta_zero_fit, Tc_fit),
                    linestyle="--",
                    color=color_cycle[b_idx % len(color_cycle)],
                    linewidth=2,
                    label=rf"$\Delta_0$={delta_zero_fit:.2f}, $T_c$={Tc_fit:.1f} (K)",
                )
            except Exception as e:
                print(f"Branch {b_idx + 1} fitting failed: {e}")

        # ax.legend(loc="upper right")

    if destpath:
        plt.savefig(destpath, dpi=300)
