# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""figexp -- recursive training of real (small) language models.

Reads ../../experiments/results_{origin,moby}{,_extra}.json (word-trigram models,
Darwin and Melville, 12 generations, 5 seeds).
(a, b) Rare-word coverage relative to generation 0, per condition, with Rogers'
       chain (dashed) whose single parameter u is fitted to the all-synthetic
       condition only and then used, unchanged, for the fresh-data conditions.
(c)    Every condition at generation 12 on two axes: rare-word coverage and the
       factor by which perplexity on held-out human text has grown. Fresh data
       is the only intervention that improves both.
Writes figexp.pdf and figexp_summary.json.
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


def load(name):
    R = json.load(open(os.path.join(EXP, f"results_{name}.json")))["results"]
    R.update(json.load(open(os.path.join(EXP, f"results_{name}_extra.json")))["results"])
    out = {}
    for c, runs in R.items():
        t = np.array([[g["tail"] for g in r] for r in runs])
        p = np.array([[g["ppl"] for g in r] for r in runs])
        out[c] = dict(rel=(t.T / t[:, 0]).T.mean(0), ppl=(p[:, -1] / p[:, 0]).mean())
    return out


def chain(u, rho, G):
    q = [1.0]
    for _ in range(G):
        q.append((1 - rho) + rho * (1 - u) * q[-1])
    return np.array(q)


STYLE = {
    "replace": ("all synthetic", RED, "-"),
    "fresh0.1": ("10% fresh", "#d88a85", "-"),
    "fresh0.3": ("30% fresh", "#9cb8d6", "-"),
    "fresh0.5": ("50% fresh", ACCENT, "-"),
    "accumulate": ("keep all past data", INK, "--"),
    "self_filter": ("filter: own likelihood", "#8c6d31", ":"),
    "ref_filter": ("filter: reference likelihood", "#6b4c9a", ":"),
    "echo_filter": ("filter: n-gram match", MUTED, ":"),
    "random_filter": ("filter: random half", "#bbbbbb", "-."),
    "novelty_filter": ("filter: favour rare words", "#3a8f6a", "-."),
}

D = {n: load(n) for n in ("origin", "moby")}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
summary = {}
for ax, (name, title) in zip(axes[:2], (("origin", "a  Darwin"), ("moby", "b  Melville"))):
    d = D[name]
    G = len(d["replace"]["rel"]) - 1
    t = np.arange(G + 1)
    y = np.log(d["replace"]["rel"][1:])
    u = 1 - np.exp(np.sum(t[1:] * y) / np.sum(t[1:] ** 2))
    summary[name] = {"u": float(u)}
    for c, (lab, col, ls) in STYLE.items():
        ax.plot(t, d[c]["rel"], color=col, ls=ls, lw=1.8, label=lab)
    for c, rho in (("replace", 1.0), ("fresh0.1", 0.9), ("fresh0.3", 0.7), ("fresh0.5", 0.5)):
        pred = chain(u, rho, G)
        ax.plot(t, pred, color=STYLE[c][1], lw=0.9, ls=(0, (2, 2)))
        summary[name][c] = {"observed": float(d[c]["rel"][-1]), "predicted": float(pred[-1])}
    ax.set_title(f"{title}: rare knowledge", fontsize=9.2, loc="left", color=INK)
    ax.set_xlabel("model generation", fontsize=9)
    ax.set_ylim(-0.03, 1.08)
    despine(ax)
axes[0].set_ylabel("rare words still present\n(relative to generation 0)", fontsize=9)

ax = axes[2]
for name, mk in (("origin", "o"), ("moby", "s")):
    for c, (lab, col, ls) in STYLE.items():
        ax.plot(D[name][c]["ppl"], D[name][c]["rel"][-1], mk, color=col, ms=6,
                mec="white", mew=0.6)
ax.set_xscale("log")
ax.set_xlabel("perplexity on human text, growth factor", fontsize=9)
ax.set_ylabel("rare-word coverage at generation 12", fontsize=9)
ax.set_title("c  only fresh data improves both", fontsize=9.2, loc="left", color=INK)
ax.annotate("better", xy=(1.3, 0.97), xytext=(3.5, 0.62), fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax.annotate("circles: Darwin\nsquares: Melville", (25, 0.85), fontsize=7.4, color=MUTED)
despine(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.6, frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig("figexp.pdf")
json.dump(summary, open("figexp_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
