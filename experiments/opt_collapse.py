# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Copyright (c) 2026 Nam H. Le. Released under the MIT License (see LICENSE).
# Paper: Le, N. H. (2026). Who Pays for the Knowledge Models Copy?
#        Model Collapse as Rogers' Paradox. Please cite it if you use this code.
"""Recursive fine-tuning of a pretrained language model (OPT-125m, wikitext-2).

Follows the setting of Shumailov et al. (2024): every generation fine-tunes the
original pretrained OPT-125m on a corpus of 64-token blocks, then generates the
next generation's synthetic blocks. Generation 0 is fine-tuned on human text
only. Synthetic blocks are sampled unconditionally (from the start token,
temperature 1), so no human text enters through prompts.

Conditions:
  replace        all synthetic
  freshF         a fraction F of the blocks fresh from the human pool (F = 0.1, 0.3, 0.5)
  self_filter    sample 2N blocks, keep the N the current model finds most likely
  random_filter  sample 2N blocks, keep a random N

Measures per generation: perplexity on the wikitext-2 test split; rare-word
coverage of the training corpus (words outside the pool's 1,000 most frequent);
distinct word trigrams per 1,000 words of the model's own samples.

Usage: python opt_collapse.py COND SEED  -> writes opt/<COND>_seed<SEED>.json
"""

import json
import math
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, BLOCK, N_BLOCKS, GENS = "facebook/opt-125m", 64, 4000, int(os.environ.get("GENS", 10))
EPOCHS, LR, BATCH, GEN_BATCH, EVAL_BLOCKS = 2, 5e-5, 32, 500, 600
DEV = "cuda"
torch.set_num_threads(2)  # keep the Windows desktop responsive


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def blocks_from(texts, tok):
    ids = tok("\n".join(texts), add_special_tokens=False)["input_ids"]
    return [ids[i:i + BLOCK] for i in range(0, len(ids) - BLOCK + 1, BLOCK)]


def batchify(blocks, bos, pad):
    L = max(len(b) for b in blocks) + 1
    x = torch.full((len(blocks), L), pad, dtype=torch.long)
    att = torch.zeros_like(x)
    for i, b in enumerate(blocks):
        x[i, :len(b) + 1] = torch.tensor([bos] + b)
        att[i, :len(b) + 1] = 1
    y = x.clone()
    y[att == 0] = -100
    y[:, 0] = -100
    return x.to(DEV), att.to(DEV), y.to(DEV)


def train(blocks, tok, seed):
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    m.train()
    order = list(range(len(blocks)))
    rng = random.Random(seed)
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for i in range(0, len(order), BATCH):
            x, att, y = batchify([blocks[j] for j in order[i:i + BATCH]], tok.bos_token_id, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(input_ids=x, attention_mask=att, labels=y).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def block_nll(m, blocks, tok):
    """Mean per-token negative log-likelihood of each block."""
    out = []
    for i in range(0, len(blocks), 256):
        x, att, y = batchify(blocks[i:i + 256], tok.bos_token_id, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(input_ids=x, attention_mask=att).logits.float()
        lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), y[:, 1:],
                                               reduction="none", ignore_index=-100)
        n = (y[:, 1:] != -100).sum(1).clamp(min=1)
        out += (lp.sum(1) / n).tolist()
    return out


@torch.no_grad()
def perplexity(m, blocks, tok):
    tot, n = 0.0, 0
    for i in range(0, len(blocks), 256):
        x, att, y = batchify(blocks[i:i + 256], tok.bos_token_id, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(input_ids=x, attention_mask=att).logits.float()
        tot += torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), y[:, 1:],
                                                 reduction="sum", ignore_index=-100).item()
        n += (y[:, 1:] != -100).sum().item()
    return math.exp(tot / n)


@torch.no_grad()
def sample(m, n, tok, seed):
    torch.manual_seed(seed)
    out = []
    while len(out) < n:
        b = min(GEN_BATCH, n - len(out))
        x = torch.full((b, 1), tok.bos_token_id, dtype=torch.long, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = m.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0,
                           top_k=0, max_new_tokens=BLOCK, min_new_tokens=BLOCK,
                           pad_token_id=tok.pad_token_id)
        out += [r[1:].tolist() for r in g]
    return out


def main(cond, seed):
    if os.path.exists(f"opt/{cond}_seed{seed}.json"):
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    keep = lambda t: t.strip() and not t.strip().startswith("=")
    pool_txt = [t.strip() for t in ds["train"]["text"] if keep(t)]
    test_txt = [t.strip() for t in ds["test"]["text"] if keep(t)]
    pool = blocks_from(pool_txt, tok)
    test = blocks_from(test_txt, tok)[:EVAL_BLOCKS]
    freq = Counter(w for t in pool_txt for w in words(t))
    rare = {w for w, _ in freq.most_common()[1000:]}
    rng = random.Random(100 + seed)

    def fresh(k):
        return [pool[i] for i in rng.sample(range(len(pool)), int(k))]

    corpus, traj = fresh(N_BLOCKS), []
    for g in range(GENS + 1):
        t0 = time.time()
        m = train(corpus, tok, seed * 100 + g)
        own = sample(m, 1000, tok, seed * 1000 + g)
        ow = words(tok.decode(sum(own, [])))
        tri = {tuple(ow[i:i + 3]) for i in range(len(ow) - 2)}
        cw = set(words(tok.decode(sum(corpus, []))))
        traj.append(dict(gen=g, ppl=perplexity(m, test, tok), tail=len(rare & cw) / len(rare),
                         div=1000 * len(tri) / max(1, len(ow))))
        print(cond, seed, traj[-1], f"{time.time() - t0:.0f}s", flush=True)
        if g == GENS:
            break
        s0 = seed * 1000 + 100 + g
        if cond == "replace":
            corpus = sample(m, N_BLOCKS, tok, s0)
        elif cond.startswith("fresh"):
            f = float(cond[5:])
            corpus = fresh(f * N_BLOCKS) + sample(m, N_BLOCKS - int(f * N_BLOCKS), tok, s0)
        elif cond in ("self_filter", "random_filter"):
            cand = sample(m, 2 * N_BLOCKS, tok, s0)
            if cond == "self_filter":
                nll = block_nll(m, cand, tok)
                cand = [cand[i] for i in np.argsort(nll)]
            else:
                rng.shuffle(cand)
            corpus = cand[:N_BLOCKS]
        del m
        torch.cuda.empty_cache()
    os.makedirs("opt", exist_ok=True)
    json.dump(traj, open(f"opt/{cond}_seed{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "all":  # run every job in opt_jobs.txt in this one process
        for line in open("opt_jobs.txt"):
            c, k = line.split()
            main(c, int(k))
        print("ALL_DONE", flush=True)
    else:
        main(sys.argv[1], int(sys.argv[2]))
