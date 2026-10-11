# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""bookstyle -- the book's fixed visual vocabulary, shared by all figures.

Direct labels, not per-figure legends: identity is carried by a
consistent role->style mapping the reader learns once and reuses
across chapters (the book-level legend). Grayscale-safe: roles differ
in lightness AND line style, never in hue alone.

Rule: direct labels up to 3 well-separated series; a legend only when
a panel has >= 4 series or heavy crossings.
"""

INK = "#2b2b2b"        # outcome / winner (correct alleles, population mean)
ACCENT = "#4878a8"     # the layer under discussion (copiers, plastic, SL)
MUTED = "#8a8a8a"      # baseline / reference (individual learners, no-learning)

ROLE = {
    "baseline": dict(color=MUTED, lw=2, ls="--"),
    "layer":    dict(color=ACCENT, lw=2, ls="-"),
    "outcome":  dict(color=INK, lw=2, ls="-"),
}


def despine(ax):
    """House axes: open top/right, muted spines and ticks, ink labels."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
