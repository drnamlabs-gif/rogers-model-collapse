# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""figopt -- recursive fine-tuning of a pretrained language model (OPT-125m).

Reads ../../experiments/opt/<cond>_seed<k>.json (OPT-125m fine-tuned afresh each
generation on 4,000 blocks of 64 tokens of wikitext-2; 10 generations; 5 seeds).
(a) Rare-word coverage relative to generation 0; dashed: two-class Rogers'
    chain (a share w of fragile words with copying loss u1, the rest with u2),
    fitted to the all-synthetic condition only and applied unchanged to the
    fresh-data conditions. The one-parameter fit is also saved in the summary.
(b) Perplexity on the wikitext-2 test split.
(c) Distinct word trigrams per 1,000 words of the model's own samples.
Writes figopt.pdf and figopt_summary.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from bookstyle import INK, ACCENT, MUTED, despine

RED = "#c0504a"
EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "opt")
CONDS = [("replace", "all synthetic", RED, "-", 1.0), ("fresh0.1", "10% fresh", "#d88a85", "-", 0.9),
         ("fresh0.3", "30% fresh", "#9cb8d6", "-", 0.7), ("fresh0.5", "50% fresh", ACCENT, "-", 0.5),
         ("random_filter", "filter: random half", MUTED, "-.", None),
         ("self_filter", "filter: own likelihood", "#8c6d31", ":", None)]
D = {c[0]: [json.load(open(os.path.join(EXP, f"{c[0]}_seed{s}.json"))) for s in range(5)] for c in CONDS}


def arr(c, k):
    return np.array([[g[k] for g in r] for r in D[c]])


def chain(u, rho, G):
    q = [1.0]
    for _ in range(G):
        q.append((1 - rho) + rho * (1 - u) * q[-1])
    return np.array(q)


rel = {c: (arr(c, "tail").T / arr(c, "tail")[:, 0]).T for c, *_ in CONDS}
G = rel["replace"].shape[1] - 1
t = np.arange(G + 1)
y = np.log(rel["replace"].mean(0)[1:])
u = 1 - np.exp(np.sum(t[1:] * y) / np.sum(t[1:] ** 2))



def two(p, rho):
    return p[0] * chain(p[1], rho, G) + (1 - p[0]) * chain(p[2], rho, G)


y = rel["replace"].mean(0)
p2 = minimize(lambda p: np.sum((np.log(two(p, 1.0)[1:]) - np.log(y[1:])) ** 2), [0.3, 0.8, 0.03],
              bounds=[(0, 0.999)] * 3).x  # least squares on log coverage, as in chain_fit.py

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
summ = {"u_one_class": float(u), "w_u1_u2": p2.tolist()}
for c, lab, col, ls, rho in CONDS:
    cov, ppl, div = rel[c], arr(c, "ppl"), arr(c, "div")
    for ax, v in zip(axes, (cov, ppl, div)):
        ax.errorbar(t, v.mean(0), yerr=v.std(0, ddof=1), color=col, ls=ls, lw=1.8, capsize=1.5, label=lab)
    s = dict(cov=cov.mean(0).tolist(), cov_sd=cov.std(0, ddof=1).tolist(),
             ppl=ppl.mean(0).tolist(), div=div.mean(0).tolist())
    if rho is not None:
        q = two(p2, rho)
        axes[0].plot(t, q, color=col, ls="--", lw=0.9)
        s["pred"], s["pred_one_class"] = q.tolist(), chain(u, rho, G).tolist()
    summ[c] = s
axes[0].set_ylabel("rare-word coverage\n(relative to generation 0)", fontsize=9)
axes[0].set_title("a  rare words; dashed: Rogers' chain", fontsize=9.2, loc="left", color=INK)
axes[0].set_ylim(0, 1.03)
axes[1].set_ylabel("perplexity (held-out human text)", fontsize=9)
axes[1].set_title("b  error on human text", fontsize=9.2, loc="left", color=INK)
axes[2].set_ylabel("distinct trigrams per 1,000 words", fontsize=9)
axes[2].set_title("c  variety of the model's output", fontsize=9.2, loc="left", color=INK)
axes[1].legend(fontsize=7.2, frameon=False, loc="upper left")
for ax in axes:
    ax.set_xlabel("generation", fontsize=9)
    despine(ax)
fig.tight_layout()
fig.savefig("figopt.pdf")
fig.savefig("figopt.png", dpi=300)
json.dump(summ, open("figopt_summary.json", "w"), indent=1)
print("ok", u, p2)
