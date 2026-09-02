"""Shared figure style for journal-ready plots."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).resolve().parent / "figures"

JOURNAL_RC = {
    "figure.figsize": (7.16, 4.4),
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    #"font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_journal_style() -> Path:
    """Apply compact publication defaults and return the figure directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(JOURNAL_RC)
    return FIGURE_DIR
