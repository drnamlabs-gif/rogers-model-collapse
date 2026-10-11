# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""Recursive fine-tuning of OPT-125m in a world that keeps changing.

The 'world' at generation t is a window of WIN consecutive 64-token blocks of
the wikitext-2 training text (Wikipedia articles in their original order,
wrapping around), shifted forward by SPEED blocks per generation, so new
articles, and their vocabulary, keep arriving. Every tenth block of the window
is held out for evaluation; fresh data are drawn from the rest. Each generation
fine-tunes the pretrained OPT-125m afresh on N_BLOCKS blocks, a fraction F fresh
from the current window and the rest sampled from the previous generation's
model (as in opt_collapse.py).

Measures per generation: coverage of the current window's rare words (outside
its 1,000 most frequent) in the training corpus, divided by the same coverage
for N_BLOCKS blocks drawn entirely fresh from the current window; perplexity on
the window's held-out blocks.

Usage: python opt_drift.py SPEED F SEED  -> writes opt_drift/s<SPEED>_f<F>_seed<SEED>.json
       python opt_drift.py all           -> runs every line of opt_drift_jobs.txt
"""

import json
import os
import random
import sys
import time
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from opt_collapse import BLOCK, EVAL_BLOCKS, GENS, MODEL, N_BLOCKS, blocks_from, perplexity, sample, train, words
import torch

WIN = 6000


def main(speed, f, seed):
    out = f"opt_drift/s{speed}_f{f}_seed{seed}.json"
    if os.path.exists(out):
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    keep = lambda t: t.strip() and not t.strip().startswith("=")
    text = blocks_from([t.strip() for t in ds["train"]["text"] if keep(t)], tok)
    L = len(text)
    rng = random.Random(700 + seed)
    start0 = rng.randrange(L)

    def window(t):
        idx = [(start0 + t * speed + i) % L for i in range(WIN)]
        return [text[i] for i in idx if i % 10], [text[i] for i in idx if i % 10 == 0][:EVAL_BLOCKS]

    def cover(blocks, rare):
        return len(rare & set(words(tok.decode(sum(blocks, []))))) / len(rare)

    trn, _ = window(0)
    corpus, traj = rng.sample(trn, N_BLOCKS), []
    for g in range(GENS + 1):
        t0 = time.time()
        trn, tst = window(g)
        freq = Counter(w for b in trn for w in words(tok.decode(b)))
        rare = {w for w, _ in freq.most_common()[1000:]}
        ref = cover(rng.sample(trn, N_BLOCKS), rare)
        m = train(corpus, tok, seed * 100 + g)
        traj.append(dict(gen=g, cov=cover(corpus, rare) / ref, ppl=perplexity(m, tst, tok)))
        print(speed, f, seed, traj[-1], f"{time.time() - t0:.0f}s", flush=True)
        if g == GENS:
            break
        nxt, _ = window(g + 1)
        k = int(f * N_BLOCKS)
        corpus = rng.sample(nxt, k) + sample(m, N_BLOCKS - k, tok, seed * 1000 + 100 + g)
        del m
        torch.cuda.empty_cache()
    os.makedirs("opt_drift", exist_ok=True)
    json.dump(dict(speed=speed, f=f, seed=seed, win=WIN, traj=traj), open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        for line in open("opt_drift_jobs.txt"):
            s, f, k = line.split()
            main(int(s), float(f), int(k))
        print("ALL_DONE", flush=True)
    else:
        main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
