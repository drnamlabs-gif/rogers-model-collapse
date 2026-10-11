# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""Summarise the OPT-125m drift sweep (opt_drift/*.json).

Per (speed, fresh fraction): mean over generations 6-10 of relative rare-word
coverage and of perplexity on the current window's held-out blocks; fastest
drift versus a static world compared per seed (Welch, Mann-Whitney, Cohen's d,
bootstrap 95% CI). Also measures the coverage that the fresh fraction alone
would give (no synthetic part), to separate what the synthetic part carries.
Writes opt_drift_summary.json.
"""

import glob
import json
import random
from collections import Counter

import numpy as np
from scipy import stats


def compare(a, b):
    a, b = np.array(a), np.array(b)
    d = a.mean() - b.mean()
    rng = np.random.default_rng(0)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
    return dict(diff=float(d), welch_p=float(stats.ttest_ind(a, b, equal_var=False).pvalue),
                mw_p=float(stats.mannwhitneyu(a, b).pvalue),
                cohen_d=float(d / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def fresh_only():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from opt_collapse import MODEL, N_BLOCKS, blocks_from, words
    from opt_drift import WIN
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    keep = lambda t: t.strip() and not t.strip().startswith("=")
    text = blocks_from([t.strip() for t in ds["train"]["text"] if keep(t)], tok)
    out = {f: [] for f in (0.1, 0.3, 0.5)}
    for seed in range(5):
        rng = random.Random(seed)
        st = rng.randrange(len(text))
        idx = [(st + i) % len(text) for i in range(WIN)]
        trn = [text[i] for i in idx if i % 10]
        freq = Counter(w for b in trn for w in words(tok.decode(b)))
        rare = {w for w, _ in freq.most_common()[1000:]}
        cov = lambda n: len(rare & set(words(tok.decode(sum(rng.sample(trn, n), []))))) / len(rare)
        ref = cov(N_BLOCKS)
        for f in out:
            out[f].append(cov(int(f * N_BLOCKS)) / ref)
    return {str(f): float(np.mean(v)) for f, v in out.items()}


D = {}
for p in glob.glob("opt_drift/*.json"):
    d = json.load(open(p))
    t = d["traj"][6:]
    D.setdefault((d["speed"], d["f"]), []).append((np.mean([g["cov"] for g in t]), np.mean([g["ppl"] for g in t])))
summ = {"cells": {}, "fast_vs_static": {}}
for (s, f), v in sorted(D.items()):
    a = np.array(v)
    summ["cells"][f"s{s}_f{f}"] = dict(n=len(a), cov=float(a[:, 0].mean()), cov_sd=float(a[:, 0].std(ddof=1)),
                                       ppl=float(a[:, 1].mean()), ppl_sd=float(a[:, 1].std(ddof=1)))
    print(f"speed {s:5d} f {f}: cov {a[:, 0].mean():.3f}  ppl {a[:, 1].mean():.1f}  (n={len(a)})")
fast = max(s for s, _ in D)
for f in sorted({f for _, f in D}):
    for i, k in ((0, "cov"), (1, "ppl")):
        r = compare([x[i] for x in D[(fast, f)]], [x[i] for x in D[(0, f)]])
        summ["fast_vs_static"][f"f{f}_{k}"] = r
        print(f"f={f} {k}: fast-static {r['diff']:+.3f}  Welch p={r['welch_p']:.2g}  MW p={r['mw_p']:.2g}  "
              f"d={r['cohen_d']:.2f}  CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
summ["fresh_only_coverage"] = fresh_only()
print("fresh fraction alone:", summ["fresh_only_coverage"])
json.dump(summ, open("opt_drift_summary.json", "w"), indent=1)
