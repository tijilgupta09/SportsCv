"""Shared dark-theme matplotlib figure setup factored out of the duplicated per-sport dashboard code.

DESIGN.md Section 6 and TECHSPEC.md Section 4.8 mandate this module so that both
football_analytics.py and cricket_analytics.py don't duplicate the same boilerplate
dark-theme matplotlib configuration (facecolor, spine colors, tick colors, etc.).

Usage
-----
    from common.dashboard_utils import make_dark_figure, apply_dark_theme

    fig, axes = make_dark_figure(nrows=2, ncols=2, figsize=(12, 8))
    # ... build subplots ...
    apply_dark_theme(axes)
"""

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import; harmless if already set
import matplotlib.pyplot as plt

# Project-wide dark-theme color constants.  Import these instead of hard-coding
# hex strings in every save_dashboard / save_pitch_map call.
DARK_BG    = "#080c12"  # figure/axes background
DARK_PANEL = "#1a2840"  # cell / patch fill
DARK_EDGE  = "#2a3860"  # cell / patch border
DARK_LABEL = "#6a8aaa"  # axis labels, ticks, minor annotations
ACCENT_CYAN = "#00e0ff"  # suptitle / primary accent
ACCENT_BLUE = "#00c8ff"  # subplot titles
DARK_HEADER = "#0a1020"  # table header row


def make_dark_figure(nrows=1, ncols=1, figsize=(10, 7), **kwargs):
    """Create a matplotlib Figure with the project dark-theme background applied.

    Returns (fig, axes) where axes is a numpy array of Axes (same shape as
    plt.subplots would return).  Pass polar=True via subplot_kw for polar plots.

    All subsequent axis-level styling (spines, ticks, labels) should be applied
    by calling apply_dark_theme(axes) after the subplots are populated.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=DARK_BG, **kwargs)
    return fig, axes


def apply_dark_theme(axes):
    """Apply the standard dark-theme style to one or more Axes objects.

    Works on a single Axes, a list, or a 2-D numpy array of Axes (the return
    value of plt.subplots with nrows>1 and ncols>1).
    """
    import numpy as np
    if isinstance(axes, np.ndarray):
        flat = axes.flatten()
    elif hasattr(axes, "__iter__"):
        flat = list(axes)
    else:
        flat = [axes]

    for ax in flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=DARK_LABEL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_EDGE)
        ax.xaxis.label.set_color(DARK_LABEL)
        ax.yaxis.label.set_color(DARK_LABEL)


def style_table(table, header_color=DARK_HEADER, cell_color=DARK_PANEL,
                edge_color=DARK_EDGE):
    """Apply the standard dark-theme style to a matplotlib Table object.

    Parameters
    ----------
    table : matplotlib.table.Table
        The table returned by ax.table(...).
    header_color : str
        Background color for the header row (row index 0).
    cell_color : str
        Background color for data rows (row index > 0).
    edge_color : str
        Border color for all cells.
    """
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=ACCENT_BLUE)
        else:
            cell.set_facecolor(cell_color)
            cell.set_text_props(color="white")
        cell.set_edgecolor(edge_color)
