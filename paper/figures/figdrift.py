# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""figdrift -- recursive training in a world that keeps changing.

Reads ../../experiments/results_drift2.json. The world is a window of 3,000
consecutive sentences sliding through the concatenated text of Darwin's
Origin and Melville's Moby-Dick, shifted by a fixed number of sentences each
generation, so new vocabulary keeps arriving. Coverage: the current world's
rare words present in the training corpus, relative to a fully fresh corpus
of the same size; averaged over generations 6-12 and 3 seeds.
(a) Coverage against the share of the world replaced per generation, one line
    per fresh fraction.
(b) Fresh fraction needed to keep 80% of fully fresh coverage, interpolated.
Writes figdrift.pdf and figdrift_summary.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bookstyle import INK, ACCENT, MUTED, despine

RED = "#c0504a"
EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments")
D = json.load(open(os.path.join(EXP, "results_drift2.json")))
R, SPEEDS, FRESH, WIN = D["results"], D["speeds"], D["fresh"], D["window"]

late = {}
sd = {}
for sp in SPEEDS:
    for f in FRESH:
        vals = [np.mean([t["coverage"] for t in r[6:]]) for r in R[f"{sp}|{f}"]]
        late[(sp, f)] = float(np.mean(vals))
        sd[(sp, f)] = float(np.std(vals))
x = [min(1.0, sp / WIN) for sp in SPEEDS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
for f, col in zip(FRESH, [RED, "#9cb8d6", ACCENT]):
    ax1.errorbar(x, [late[(sp, f)] for sp in SPEEDS], yerr=[sd[(sp, f)] for sp in SPEEDS],
                 color=col, marker="o", ms=5, lw=1.8, capsize=2, label=f"{int(f * 100)}% fresh")
ax1.set_xlabel("share of the world replaced per generation", fontsize=9)
ax1.set_ylabel("rare-word coverage\n(relative to fully fresh)", fontsize=9)
ax1.set_title("a  a moving world erodes what copies carry", fontsize=9.2, loc="left", color=INK)
ax1.legend(fontsize=7.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36))
ax1.set_ylim(0.35, 1.0)

need = {}
for sp in SPEEDS:
    cov = [late[(sp, f)] for f in FRESH]
    need[sp] = float(np.interp(0.8, cov, FRESH))
ax2.plot(x, [need[sp] for sp in SPEEDS], "o-", color=INK, lw=1.8, ms=5)
ax2.set_xlabel("share of the world replaced per generation", fontsize=9)
ax2.set_ylabel("fresh fraction needed to keep\n80% of fully fresh coverage", fontsize=9)
ax2.set_title("b  the price of a moving world", fontsize=9.2, loc="left", color=INK)
ax2.set_ylim(0, 0.6)
for ax in (ax1, ax2):
    despine(ax)
fig.tight_layout()
fig.savefig("figdrift.pdf")
summ = dict(coverage={f"{sp}|{f}": late[(sp, f)] for sp in SPEEDS for f in FRESH},
            sd={f"{sp}|{f}": sd[(sp, f)] for sp in SPEEDS for f in FRESH},
            need80={str(sp): need[sp] for sp in SPEEDS}, window=WIN)
json.dump(summ, open("figdrift_summary.json", "w"), indent=1)
print(json.dumps(summ["need80"], indent=1))
