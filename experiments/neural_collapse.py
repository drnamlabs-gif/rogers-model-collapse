# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""Recursive training of a small character-level transformer (Darwin corpus).

Same design as collapse_filters.py, with a neural language model in place of the
trigram model. Each generation trains a fresh 2-layer transformer (d=128, 4 heads,
context 128 characters) on a corpus of about N characters, then samples the next
corpus from it (temperature 1). Conditions: replace, fresh0.3, accumulate,
self_filter (keep the half of sampled lines the current model finds most likely),
random_filter (keep a random half). Measures per generation: bits per character
on held-out human text; rare-word coverage of the training corpus; distinct word
trigrams per 1,000 generated words.

Usage: python neural_collapse.py COND SEED  -> writes neural/<COND>_seed<SEED>.json
"""

import json
import math
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
N_CHARS, GENS, STEPS, BATCH, CTX = 60_000, 6, 1000, 32, 128


def load_lines():
    raw = open("data/origin.txt", encoding="utf8").read()
    a, b = raw.index("*** START OF"), raw.index("*** END OF")
    text = raw[raw.index("\n", a):b].lower()
    text = re.sub(r"[^a-z.,;:!?' \n]", " ", text)
    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"(?<=[.!?])\s+", text)]
    return [s for s in sents if 20 <= len(s) <= 400]


CHARS = sorted(set("abcdefghijklmnopqrstuvwxyz.,;:!?' \n"))
STOI = {c: i for i, c in enumerate(CHARS)}


def enc(text):
    return torch.tensor([STOI.get(c, STOI[" "]) for c in text], dtype=torch.long)


class TinyLM(nn.Module):
    def __init__(self, v=len(CHARS), d=128, heads=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.pos = nn.Embedding(CTX, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, v)
        self.register_buffer("mask", torch.triu(torch.full((CTX, CTX), float("-inf")), 1))

    def forward(self, x):
        t = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(t))
        h = self.tr(h, mask=self.mask[:t, :t], is_causal=True)
        return self.out(h)


def train(corpus_lines, seed):
    torch.manual_seed(seed)
    data = enc("\n".join(corpus_lines) + "\n")
    m = TinyLM()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    m.train()
    for step in range(STEPS):
        ix = torch.randint(0, len(data) - CTX - 1, (BATCH,))
        x = torch.stack([data[i:i + CTX] for i in ix])
        y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
        loss = nn.functional.cross_entropy(m(x).reshape(-1, len(CHARS)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m


@torch.no_grad()
def bpc(m, lines):
    data = enc("\n".join(lines) + "\n")
    tot, n = 0.0, 0
    for i in range(0, len(data) - CTX - 1, CTX):
        x = data[i:i + CTX][None]
        y = data[i + 1:i + CTX + 1][None]
        tot += nn.functional.cross_entropy(m(x).reshape(-1, len(CHARS)), y.reshape(-1),
                                           reduction="sum").item()
        n += y.numel()
    return tot / n / math.log(2)


@torch.no_grad()
def line_nll(m, line):
    data = enc("\n" + line + "\n")[: CTX + 1]
    x, y = data[:-1][None], data[1:][None]
    return nn.functional.cross_entropy(m(x).reshape(-1, len(CHARS)), y.reshape(-1)).item()


@torch.no_grad()
def sample(m, n_chars, gen, streams=256):
    """Sample lines from `streams` parallel character streams until n_chars."""
    out, total = [], 0
    ctx = torch.full((streams, 1), STOI["\n"], dtype=torch.long)
    bufs = [[] for _ in range(streams)]
    while total < n_chars:
        logits = m(ctx[:, -CTX:])[:, -1]
        c = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)
        ctx = torch.cat([ctx, c], 1)[:, -CTX:]
        for k, ci in enumerate(c[:, 0].tolist()):
            ch = CHARS[ci]
            if ch == "\n":
                line = "".join(bufs[k]).strip()
                if len(line) >= 20:
                    out.append(line)
                    total += len(line)
                bufs[k] = []
            else:
                bufs[k].append(ch)
                if len(bufs[k]) > 400:
                    bufs[k] = []
    return out


def take(lines, n_chars, rng):
    pool = lines[:]
    rng.shuffle(pool)
    out, n = [], 0
    for s in pool:
        if n >= n_chars:
            break
        out.append(s)
        n += len(s)
    return out


def words(lines):
    return [w for s in lines for w in re.findall(r"[a-z]+", s)]


def main(cond, seed):
    lines = load_lines()
    random.Random(0).shuffle(lines)
    n = len(lines)
    test, pool = lines[: n // 10], lines[n // 10:]
    freq = Counter(words(pool))
    rare = {w for w, _ in freq.most_common()[1000:]}
    rng = random.Random(100 + seed)
    gen = torch.Generator().manual_seed(seed)
    human0 = take(pool, N_CHARS, rng)
    corpus, all_synth, traj = human0, [], []
    for g in range(GENS + 1):
        m = train(corpus, seed * 100 + g)
        own = sample(m, 20_000, gen)
        ow = words(own)
        tri = {tuple(ow[i:i + 3]) for i in range(len(ow) - 2)}
        traj.append(dict(gen=g, bpc=bpc(m, test[:400]),
                         tail=len(rare & set(words(corpus))) / len(rare),
                         div=1000 * len(tri) / max(1, len(ow))))
        print(cond, seed, traj[-1], flush=True)
        if g == GENS:
            break
        if cond == "replace":
            corpus = sample(m, N_CHARS, gen)
        elif cond == "fresh0.3":
            corpus = take(pool, 0.3 * N_CHARS, rng) + sample(m, 0.7 * N_CHARS, gen)
        elif cond == "accumulate":
            all_synth += sample(m, N_CHARS, gen)
            corpus = human0 + all_synth
        elif cond in ("self_filter", "random_filter"):
            cand = sample(m, 2 * N_CHARS, gen)
            if cond == "self_filter":
                cand.sort(key=lambda s: line_nll(m, s))
            else:
                rng.shuffle(cand)
            corpus, k = [], 0
            for s in cand:
                if k >= N_CHARS:
                    break
                corpus.append(s)
                k += len(s)
    os.makedirs("neural", exist_ok=True)
    json.dump(traj, open(f"neural/{cond}_seed{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
