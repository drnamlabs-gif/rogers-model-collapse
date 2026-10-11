# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""Recursive training in a world that keeps changing (sustained drift).

The 'world' at generation t is a window of WIN consecutive sentences in the
concatenated text of Darwin's Origin and Melville's Moby-Dick (in reading
order, wrapping around), shifted forward by `speed` sentences per generation.
New vocabulary therefore keeps arriving. Fresh data are drawn from the current
window's training sentences; evaluation uses the current window's held-out
sentences (every tenth sentence). A lineage of word-trigram models is trained
on N tokens per generation, a fraction f fresh and the rest sampled from the
previous model.

Measure at each generation: coverage of the current window's rare words
(outside its 1,000 most frequent) in the training corpus, divided by the same
coverage for a corpus of N fully fresh tokens from the current window.
Reported: mean over generations 6-12 (after the transient).

Grid: speed in {0, 500, 1000, 2000, 3000} sentences per generation (window 3,000);
f in {0.1, 0.3, 0.5}; 12 generations; 3 seeds. Writes results_drift2.json.
"""

import json
import random
import sys
from collections import Counter

sys.path.insert(0, ".")
from collapse_filters import Trigram, load, synth  # noqa: E402

N_TOKENS, GENS, SEEDS, WIN = 30000, 12, 3, 3000
SPEEDS = [0, 500, 1000, 2000, 3000]
FRESH = [0.1, 0.3, 0.5]


def take_from(sents, n_tokens, rng):
    out, n = [], 0
    while n < n_tokens:
        s = sents[rng.randrange(len(sents))]
        out.append(s)
        n += len(s)
    return out


def main():
    text = load("origin") + load("moby")
    L = len(text)
    vocab = {w for s in text for w in s}

    def window(start):
        idx = [(start + i) % L for i in range(WIN)]
        train = [text[i] for i in idx if i % 10 != 0]
        test = [text[i] for i in idx if i % 10 == 0]
        return train, test

    res = {}
    for speed in SPEEDS:
        for f in FRESH:
            runs = []
            for seed in range(SEEDS):
                rng = random.Random(500 + seed)
                start0 = rng.randrange(L)
                train, _ = window(start0)
                corpus = take_from(train, N_TOKENS, rng)
                traj = []
                for g in range(GENS + 1):
                    train, test = window(start0 + speed * g)
                    m = Trigram(corpus, vocab)
                    fresh_corpus = take_from(train, N_TOKENS, random.Random(77 + g + seed))
                    freq = Counter(x for s in train for x in s)
                    rare = {x for x, _ in freq.most_common()[1000:]}
                    words = {x for s in corpus for x in s}
                    ref_words = {x for s in fresh_corpus for x in s}
                    cov = len(rare & words) / max(1, len(rare & ref_words))
                    traj.append(dict(gen=g, coverage=cov))
                    if g == GENS:
                        break
                    nxt, _ = window(start0 + speed * (g + 1))
                    corpus = take_from(nxt, f * N_TOKENS, rng) + synth(m, (1 - f) * N_TOKENS, rng)
                runs.append(traj)
                late = sum(t["coverage"] for t in traj[6:]) / len(traj[6:])
                print(f"speed {speed} fresh {f} seed {seed}: late coverage {late:.3f}", flush=True)
            res[f"{speed}|{f}"] = runs
    json.dump(dict(results=res, speeds=SPEEDS, fresh=FRESH, gens=GENS, window=WIN),
              open("results_drift2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
