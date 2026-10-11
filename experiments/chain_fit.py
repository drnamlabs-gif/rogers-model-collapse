# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""One-class versus two-class Rogers chain, fitted to fully synthetic training only.

For every model family (word trigram on Darwin and Melville, character
transformer, OPT-125m) the chain is fitted, by least squares on log coverage,
to the mean relative rare-word coverage of the all-synthetic condition, and
then applied with no further fitting to the fresh-data conditions.

  one class:   q_t = chain(u, rho)
  two classes: q_t = w chain(u1, rho) + (1 - w) chain(u2, rho)

Model choice uses the all-synthetic condition only: each form is fitted to the
first half of the generations and scored on the second half. Reports fitted
parameters, the held-out error, and the error of the out-of-sample predictions
(final generation, and maximum over generations). Writes chain_fit.json and the
LaTeX rows of the appendix table (chain_fit_table.tex).
"""

import json
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
RHO = {"replace": 1.0, "fresh0.1": 0.9, "fresh0.3": 0.7, "fresh0.5": 0.5}
LABEL = {"trigram_origin": "trigram, Darwin", "trigram_moby": "trigram, Melville",
         "char_transformer": "character transformer", "opt125m": "OPT-125m"}


def chain(u, rho, G):
    q = [1.0]
    for _ in range(G):
        q.append((1 - rho) + rho * (1 - u) * q[-1])
    return np.array(q)


def one(p, rho, G):
    return chain(p[0], rho, G)


def two(p, rho, G):
    return p[0] * chain(p[1], rho, G) + (1 - p[0]) * chain(p[2], rho, G)


FORMS = {"one": (one, [[0.05], [0.2], [0.5], [0.8]]),
         "two": (two, [[a, b, c] for a in (0.2, 0.5) for b in (0.5, 0.9) for c in (0.02, 0.1)])}


def rel(runs):
    a = np.array([[g["tail"] for g in r] for r in runs])
    return (a.T / a[:, 0]).T.mean(0)


def families():
    out = {}
    for name in ("origin", "moby"):
        R = json.load(open(f"results_{name}.json"))["results"]
        out[f"trigram_{name}"] = {c: rel(R[c]) for c in RHO}
    out["char_transformer"] = {c: rel([json.load(open(f"neural/{c}_seed{s}.json")) for s in range(3)])
                               for c in ("replace", "fresh0.3")}
    out["opt125m"] = {c: rel([json.load(open(f"opt/{c}_seed{s}.json")) for s in range(5)]) for c in RHO}
    return out


def fit(form, y, G, upto):
    f, starts = FORMS[form]
    loss = lambda p: np.sum((np.log(np.maximum(f(p, 1.0, G)[1:upto + 1], 1e-9)) - np.log(y[1:upto + 1])) ** 2)
    return min((minimize(loss, s, bounds=[(0, 0.999)] * len(s)) for s in starts), key=lambda r: r.fun).x


res, rows = {}, []
for fam, d in families().items():
    y = d["replace"]
    G = len(y) - 1
    h = G // 2
    res[fam] = {}
    for form, (f, _) in FORMS.items():
        holdout = float(np.abs(f(fit(form, y, G, h), 1.0, G)[h + 1:] - y[h + 1:]).max())
        p = fit(form, y, G, G)
        pred = {c: dict(obs=float(d[c][-1]), pred=float(f(p, RHO[c], G)[-1]),
                        max_err=float(np.abs(f(p, RHO[c], G) - d[c]).max())) for c in d if c != "replace"}
        res[fam][form] = dict(params=p.round(3).tolist(), holdout=holdout,
                              final_err=max(abs(v["obs"] - v["pred"]) for v in pred.values()),
                              max_err=max(v["max_err"] for v in pred.values()), pred=pred)
    pick = min(FORMS, key=lambda k: res[fam][k]["holdout"])
    res[fam]["selected"] = pick
    for form in FORMS:
        o = res[fam][form]
        par = (f"$u={o['params'][0]:.2f}$" if form == "one" else
               f"$w={o['params'][0]:.2f}$, $u_1={o['params'][1]:.2f}$, $u_2={o['params'][2]:.2f}$")
        star = "$^{*}$" if form == pick else ""
        rows.append(f"{LABEL[fam] if form == 'one' else ''} & {form}{star} & {par} & {o['holdout']:.3f} & "
                    f"{o['final_err']:.3f} & {o['max_err']:.3f} \\\\")
        print(f"{fam:18s} {form}{'*' if form == pick else ' '} {o['params']}  holdout {o['holdout']:.3f}  "
              f"final {o['final_err']:.3f}  max {o['max_err']:.3f}  "
              + str({c: (round(v['obs'], 3), round(v['pred'], 3)) for c, v in o['pred'].items()}))
    rows.append("\\addlinespace")
json.dump(res, open("chain_fit.json", "w"), indent=1)
open("chain_fit_table.tex", "w").write("\n".join(rows[:-1]) + "\n")
