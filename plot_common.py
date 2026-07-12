"""Shared presentation helpers for kinematic plots.

This module intentionally contains no experiment-specific data processing.
"""

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


def format_axes(
        axes,
        xticks=None,
        yticks=None,
        xlabel=None,
        ylabel=None,
        ylabel_size=10,
        xlabel_size=10,
        spine_width=3,
        tick_width=3
):
    """Apply the repository's standard axis formatting."""
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for axis in axes:
        axis.spines["left"].set_linewidth(spine_width)
        axis.spines["bottom"].set_linewidth(spine_width)
        axis.tick_params(axis="both", width=tick_width, length=6)

        if xticks is not None:
            axis.set_xticks(xticks)
        if yticks is not None:
            axis.set_yticks(yticks)
        if xlabel is not None:
            axis.set_xlabel(xlabel, fontsize=xlabel_size)
        if ylabel is not None:
            axis.set_ylabel(ylabel, fontsize=ylabel_size)


def centered_shades(color, n_shades=5, spread=0.6):
    """Return evenly spaced darker/lighter shades around a base color."""
    if n_shades % 2 == 0:
        raise ValueError("n_shades should be odd to center on base color.")

    base_rgb = np.array(mcolors.to_rgb(color))
    shades = []
    for factor in np.linspace(-spread, spread, n_shades):
        if factor < 0:
            new_color = base_rgb * (1 + factor)
        else:
            new_color = base_rgb + (1 - base_rgb) * factor
        shades.append(tuple(new_color))
    return shades


def significance_label(p_value):
    """Convert a p-value to the significance labels used in figures."""
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_p_value(p_value):
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def format_rho_value(rho):
    if pd.isna(rho):
        return "rho=NA"
    return f"rho={rho:.2f}"


def add_significance_bracket(axis, x1, x2, y, text):
    """Draw a significance bracket using the current y-axis scale."""
    y_range = axis.get_ylim()[1] - axis.get_ylim()[0]
    height = y_range * 0.025
    axis.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color="black",
        linewidth=1
    )
    axis.text(
        (x1 + x2) / 2,
        y + height,
        text,
        ha="center",
        va="bottom",
        fontsize=11
    )
