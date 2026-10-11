# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""figneural -- recursive training of a small character-level transformer.

Reads ../../experiments/neural/<cond>_seed<k>.json (2-layer transformer, d=128,
Darwin corpus, 6 generations, 3 seeds).
(a) Bits per character on held-out human text, per condition.
(b) Rare-word coverage of the training corpus, relative to generation 0.
Writes figneural.pdf and figneural_summary.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bookstyle import INK, ACCENT, MUTED, despine

RED = "#c0504a"
EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "neural")
CONDS = [("fresh0.3", "30% fresh", ACCENT), ("replace", "fully synthetic", INK),
         ("random_filter", "random half kept", MUTED), ("self_filter", "most likely half kept", RED)]
D = {c: [json.load(open(os.path.join(EXP, f"{c}_seed{s}.json"))) for s in range(3)] for c, _, _ in CONDS}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
summ = {}
for c, lab, col in CONDS:
    bpc = np.array([[g["bpc"] for g in r] for r in D[c]])
    cov = np.array([[g["tail"] / r[0]["tail"] for g in r] for r in D[c]])
    t = np.arange(bpc.shape[1])
    ax1.errorbar(t, bpc.mean(0), yerr=bpc.std(0, ddof=1), color=col, marker="o", ms=4,
                 lw=1.8, capsize=2, label=lab)
    ax2.errorbar(t, cov.mean(0), yerr=cov.std(0, ddof=1), color=col, marker="o", ms=4,
                 lw=1.8, capsize=2, label=lab)
    summ[c] = dict(bpc=bpc.mean(0).tolist(), bpc_sd=bpc.std(0, ddof=1).tolist(),
                   cov=cov.mean(0).tolist(), cov_sd=cov.std(0, ddof=1).tolist())
ax1.set_xlabel("generation", fontsize=9)
ax1.set_ylabel("bits per character\n(held-out human text)", fontsize=9)
ax1.set_title("a  error grows unless fresh data come in", fontsize=9.2, loc="left", color=INK)
ax1.legend(fontsize=7.6, frameon=False, loc="upper left")
ax2.set_xlabel("generation", fontsize=9)
ax2.set_ylabel("rare-word coverage\n(relative to generation 0)", fontsize=9)
ax2.set_title("b  rare words go first", fontsize=9.2, loc="left", color=INK)
ax2.set_ylim(0, 1.05)
for ax in (ax1, ax2):
    despine(ax)
fig.tight_layout()
fig.savefig("figneural.pdf")
json.dump(summ, open("figneural_summary.json", "w"), indent=1)
print("ok")
