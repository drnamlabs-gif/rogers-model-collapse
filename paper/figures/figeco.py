# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""figeco -- Rogers' paradox for a human-AI knowledge ecosystem.

Model (one generation = one model release):
  * K facts; each changes to a new value with probability u per generation.
  * N people. A share h learn for themselves (cost c, always right); the rest
    ask the current model (cost 0; right if the model holds the fact and the
    fact has not changed since the model was trained). With 'critical' asking,
    a person checks the model's answer (cost cv) and learns for themselves
    (cost c) when it is wrong.
  * The model released at generation t+1 is trained on a corpus whose share s
    is synthetic (written by the previous model) and 1-s is written by people:
    learners write the truth, askers write whatever the model told them.
  * People's strategies evolve by payoff-proportional imitation with mutation.
Analytic equilibrium (plain asking, interior): model accuracy when used
= 1-c (accuracy when trained (1-c)/(1-u)); fresh fraction of the corpus f* = u(1-c)/(c(1-u)); share of
learners h* = f*/((1-s) kappa), kappa = 1 - exp(-N/K) the share of facts
someone writes about in a generation; mean payoff = 1-c (no better than a world without the
model). Panels: (a) mean payoff after the model is introduced, plain vs
critical asking; (b) equilibrium share of learners against the synthetic share
s: formula and simulation; (c) model quality against s. Computed.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bookstyle import INK, ACCENT, MUTED, despine

RED = "#c0504a"
U, C, CV, MU = 0.05, 0.3, 0.05, 0.001
K, N = 4000, 4000


def run(s, critical, T=400, seed=0, h0=0.99):
    rng = np.random.default_rng(seed)
    world = np.zeros(K, np.int64)
    next_val = 1
    model = world.copy()
    h = h0
    pay_hist, h_hist, q_hist = [], [], []
    for t in range(T):
        flip = rng.random(K) < U
        n_flip = int(flip.sum())
        world[flip] = np.arange(next_val, next_val + n_flip)
        next_val += n_flip
        q = float((model == world).mean())
        learners = rng.random(N) < h
        facts = rng.integers(0, K, N)
        right = model[facts] == world[facts]
        pay = np.where(learners, 1.0 - C, right.astype(float))
        if critical:
            pay = np.where(learners, 1.0 - C, np.where(right, 1.0 - CV, 1.0 - CV - C))
        written = np.where(learners | (critical & ~right), world[facts], model[facts])
        # corpus for the next model: per fact, a draw from people's writing
        # (share 1-s) or from the model itself (share s)
        human_writes = np.full(K, -1, np.int64)
        human_writes[facts] = written
        from_model = (rng.random(K) < s) | (human_writes < 0)
        new_model = np.where(from_model, model, human_writes)
        # facts nobody wrote about this round fall back to the model's own text
        model = new_model
        pl = pay[learners].mean() if learners.any() else 1.0 - C
        pa = pay[~learners].mean() if (~learners).any() else pl
        hbar = learners.mean()
        mean_pay = hbar * pl + (1 - hbar) * pa
        h = h + 0.5 * h * (1 - h) * (pl - pa)
        h = (1 - MU) * h + MU * 0.5
        h = min(max(h, 0.0), 1.0)
        pay_hist.append(mean_pay)
        h_hist.append(h)
        q_hist.append(q)
    return np.array(pay_hist), np.array(h_hist), np.array(q_hist)


def analytic(s):
    q = 1 - C  # model accuracy at the time people use it
    f = U * (1 - C) / (C * (1 - U))
    kappa = 1 - np.exp(-N / K)
    h = f / ((1 - s) * kappa)
    return q, min(h, 1.0)


S = [0.0, 0.2, 0.4, 0.6, 0.8]
sims = {}
for s in S:
    p, h, q = run(s, False)
    sims[s] = dict(h=float(h[-100:].mean()), q=float(q[-100:].mean()), pay=float(p[-100:].mean()))
pp, hp, qp = run(0.0, False, T=300)
pc, hc, qc = run(0.0, True, T=300)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.2))
ax1.plot(pp, color=RED, lw=1.6, label="people ask the model")
ax1.plot(pc, color=ACCENT, lw=1.6, label="people ask, check, relearn if wrong")
ax1.axhline(1 - C, color=MUTED, ls="--", lw=1.2)
ax1.annotate("no model: 1 - c", (150, 1 - C - 0.045), fontsize=7.8, color=MUTED)
ax1.set_xlabel("model generation", fontsize=9)
ax1.set_ylabel("mean payoff per person", fontsize=9)
ax1.set_title("a  the paradox, and the rescue", fontsize=9.2, loc="left", color=INK)
ax1.legend(fontsize=7.2, frameon=False, loc="lower right")
ax1.set_ylim(0.55, 1.0)

ss = np.linspace(0, 0.9, 100)
ax2.plot(ss, [analytic(x)[1] for x in ss], color=INK, lw=1.6, ls="--", label="formula")
ax2.plot(S, [sims[x]["h"] for x in S], "o", color=RED, ms=6, label="simulation")
ax2.set_xlabel("share of the corpus written by models, s", fontsize=9)
ax2.set_ylabel("share of people who learn for themselves", fontsize=9)
ax2.set_title("b  synthetic text is paid for by people", fontsize=9.2, loc="left", color=INK)
ax2.legend(fontsize=7.4, frameon=False, loc="upper left")
ax2.set_ylim(0, 1.05)

ax3.plot(ss, [analytic(x)[0] for x in ss], color=INK, lw=1.6, ls="--", label="formula")
ax3.plot(S, [sims[x]["q"] for x in S], "o", color=RED, ms=6, label="simulation")
ax3.set_xlabel("share of the corpus written by models, s", fontsize=9)
ax3.set_ylabel("model accuracy when used", fontsize=9)
ax3.set_title("c  model quality is set by people's costs", fontsize=9.2, loc="left", color=INK)
ax3.set_ylim(0.5, 1.0)
ax3.legend(fontsize=7.4, frameon=False, loc="lower left")
for ax in (ax1, ax2, ax3):
    despine(ax)
fig.tight_layout()
fig.savefig("figeco.pdf")
out = dict(params=dict(u=U, c=C, cv=CV, mu=MU, K=K, N=N),
           analytic={str(s): analytic(s) for s in S}, sims={str(k): v for k, v in sims.items()},
           plain_mean_payoff_last100=float(pp[-100:].mean()), critical_mean_payoff_last100=float(pc[-100:].mean()))
json.dump(out, open("figeco_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
