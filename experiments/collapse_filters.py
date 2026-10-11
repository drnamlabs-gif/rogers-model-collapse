# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""Model collapse as Rogers' paradox: fresh data, and what a filter checks against.

A lineage of word-trigram language models, each trained on a corpus built from
the previous model's samples. Two corpora (Darwin, On the Origin of Species;
Melville, Moby-Dick; Project Gutenberg). Per corpus, sentences are split into:
  test   10%  -- held-out human text, used only for evaluation
  ref    20%  -- held-out human text, used only by the world-anchored filter
  pool   70%  -- the human pool from which generation 0 and fresh data are drawn

Conditions (each generation trains on N tokens):
  replace      all synthetic (rho = 1)
  fresh{f}     fraction f of fresh human sentences, the rest synthetic
  accumulate   original human corpus + every synthetic corpus so far
  self_filter  sample 2N tokens, keep the half the CURRENT model finds most
               likely (lowest per-token perplexity) -- a typicality filter
  ref_filter   sample 2N tokens, keep the half a REFERENCE model trained on
               unseen human text finds most likely -- a check against the world
  echo_filter  sample, keep only sentences every trigram of which occurs in the
               human pool -- a check against what is already known

Measures per generation: perplexity on test; rare-word coverage (share of the
pool's words outside the 1,000 most frequent that occur in the training
corpus); distinct trigrams per 1,000 tokens of the model's own samples.
Writes results_<corpus>.json.  Usage: python collapse_filters.py origin|moby
"""

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

N_TOKENS, GENS, SEEDS = 30000, 12, 5
L3, L2, L1 = 0.6, 0.3, 0.1
BOS, EOS = "<s>", "</s>"


def load(name):
    raw = open(f"data/{name}.txt", encoding="utf8").read()
    a = raw.index("*** START OF")
    b = raw.index("*** END OF")
    text = raw[raw.index("\n", a):b].lower()
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        toks = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;:!?]", s)
        if 4 <= len(toks) <= 100:
            out.append(toks)
    return out


class Trigram:
    def __init__(self, sents, vocab):
        self.vocab = vocab
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        for s in sents:
            w = [BOS, BOS] + s + [EOS]
            for i in range(2, len(w)):
                self.uni[w[i]] += 1
                self.bi[w[i - 1]][w[i]] += 1
                self.tri[(w[i - 2], w[i - 1])][w[i]] += 1
        self.n_uni = sum(self.uni.values())
        self.V = len(vocab) + 1
        self.bi_tot = {k: sum(c.values()) for k, c in self.bi.items()}
        self.tri_tot = {k: sum(c.values()) for k, c in self.tri.items()}

    def prob(self, w2, w1, w):
        p1 = (self.uni.get(w, 0) + 0.01) / (self.n_uni + 0.01 * self.V)
        c2 = self.bi.get(w1)
        p2 = c2[w] / self.bi_tot[w1] if c2 else p1
        c3 = self.tri.get((w2, w1))
        p3 = c3[w] / self.tri_tot[(w2, w1)] if c3 else p2
        return L3 * p3 + L2 * p2 + L1 * p1

    def sent_logppl(self, s):
        w = [BOS, BOS] + [x if x in self.vocab else "<unk>" for x in s] + [EOS]
        lp = sum(math.log(self.prob(w[i - 2], w[i - 1], w[i])) for i in range(2, len(w)))
        return -lp / (len(w) - 2)

    def ppl(self, sents):
        tot, n = 0.0, 0
        for s in sents:
            tot += self.sent_logppl(s) * (len(s) + 1)
            n += len(s) + 1
        return math.exp(tot / n)

    @staticmethod
    def _draw(counter, rng):
        r = rng.random() * sum(counter.values())
        for k, c in counter.items():
            r -= c
            if r <= 0:
                return k
        return k

    def sample(self, rng, max_len=100):
        w2, w1, out = BOS, BOS, []
        for _ in range(max_len):
            u = rng.random()
            c3, c2 = self.tri.get((w2, w1)), self.bi.get(w1)
            if u < L3 and c3:
                w = self._draw(c3, rng)
            elif u < L3 + L2 and c2:
                w = self._draw(c2, rng)
            else:
                w = self._draw(self.uni, rng)
            if w == EOS:
                break
            out.append(w)
            w2, w1 = w1, w
        return out


def take(sents, n_tokens, rng):
    pool = sents[:]
    rng.shuffle(pool)
    out, n = [], 0
    for s in pool:
        if n >= n_tokens:
            break
        out.append(s)
        n += len(s)
    return out


def synth(model, n_tokens, rng):
    out, n = [], 0
    while n < n_tokens:
        s = model.sample(rng)
        if len(s) >= 4:
            out.append(s)
            n += len(s)
    return out


def keep_best(cands, scorer, n_tokens):
    ranked = sorted(cands, key=scorer)
    out, n = [], 0
    for s in ranked:
        if n >= n_tokens:
            break
        out.append(s)
        n += len(s)
    return out


def tri_set(sents):
    t = set()
    for s in sents:
        w = [BOS, BOS] + s
        for i in range(2, len(w)):
            t.add((w[i - 2], w[i - 1], w[i]))
    return t


def main(name):
    sents = load(name)
    random.Random(0).shuffle(sents)
    n = len(sents)
    test, ref, pool = sents[: n // 10], sents[n // 10: 3 * n // 10], sents[3 * n // 10:]
    freq = Counter(w for s in pool for w in s)
    vocab = set(freq) | {w for s in ref for w in s}
    rare = {w for w, _ in freq.most_common()[1000:]}
    pool_tris = tri_set(pool)
    ref_model = Trigram(ref, vocab)
    print(f"{name}: {n} sentences, pool {sum(map(len, pool))} tokens, rare {len(rare)}", flush=True)

    conds = ["replace", "fresh0.1", "fresh0.3", "fresh0.5", "accumulate",
             "self_filter", "ref_filter", "echo_filter", "random_filter", "novelty_filter"]
    if len(sys.argv) > 2:
        conds = sys.argv[2].split(",")
    res = {c: [] for c in conds}
    for seed in range(SEEDS):
        rng = random.Random(100 + seed)
        human0 = take(pool, N_TOKENS, rng)
        for c in conds:
            corpus, all_synth, traj = human0, [], []
            for g in range(GENS + 1):
                m = Trigram(corpus, vocab)
                own = synth(m, 5000, rng)
                words = {w for s in corpus for w in s}
                traj.append(dict(gen=g, ppl=m.ppl(test), tail=len(rare & words) / len(rare),
                                 div=1000 * len(tri_set(own)) / sum(map(len, own))))
                if g == GENS:
                    break
                if c == "replace":
                    corpus = synth(m, N_TOKENS, rng)
                elif c.startswith("fresh"):
                    f = float(c[5:])
                    corpus = take(pool, f * N_TOKENS, rng) + synth(m, (1 - f) * N_TOKENS, rng)
                elif c == "accumulate":
                    all_synth += synth(m, N_TOKENS, rng)
                    corpus = human0 + all_synth
                elif c == "self_filter":
                    corpus = keep_best(synth(m, 2 * N_TOKENS, rng), m.sent_logppl, N_TOKENS)
                elif c == "ref_filter":
                    corpus = keep_best(synth(m, 2 * N_TOKENS, rng), ref_model.sent_logppl, N_TOKENS)
                elif c == "random_filter":
                    cand = synth(m, 2 * N_TOKENS, rng)
                    rng.shuffle(cand)
                    corpus = keep_best(cand, lambda s: 0.0, N_TOKENS)
                elif c == "novelty_filter":
                    cnt = Counter(w for s in corpus for w in s)
                    corpus = keep_best(synth(m, 2 * N_TOKENS, rng),
                                       lambda s: -sum(1.0 / (1 + cnt[w]) for w in s) / len(s), N_TOKENS)
                elif c == "echo_filter":
                    cand, out, k = synth(m, 6 * N_TOKENS, rng), [], 0
                    for s in cand:
                        w = [BOS, BOS] + s
                        if all((w[i - 2], w[i - 1], w[i]) in pool_tris for i in range(2, len(w))):
                            out.append(s)
                            k += len(s)
                            if k >= N_TOKENS:
                                break
                    corpus = out if out else cand[:1]
            res[c].append(traj)
            t0, t1 = traj[0], traj[-1]
            print(f"{name} seed {seed} {c:12s} ppl {t0['ppl']:6.0f} -> {t1['ppl']:7.0f}  "
                  f"tail {t0['tail']:.2f} -> {t1['tail']:.2f}  div {t0['div']:.0f} -> {t1['div']:.0f}",
                  flush=True)
    tag = "" if len(sys.argv) <= 2 else "_extra"
    json.dump(dict(results=res, n_tokens=N_TOKENS, gens=GENS, corpus=name),
              open(f"results_{name}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
