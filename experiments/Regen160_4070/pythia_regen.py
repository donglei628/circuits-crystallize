"""
REGEN-Pythia — can a lesioned induction circuit REGROW in a real LM? (cross-substrate test of D1/D3/regeneration)

Take a trained Pythia checkpoint, identify the induction SET S (the redundant cluster, |S|~7-9 heads — a single
head is not enough, see D-Pythia), CLAMP it to zero, then CONTINUE TRAINING on wikitext with the lesion maintained.
Track whether the induction function (ICL loss drop on a repeated-random probe) REGROWS — and WHERE (the clamped
set is frozen, so any recovery is RELOCATION into the surviving free heads = the toy-D3 prediction).

Conditions:
  CLAMP   : keep S zeroed during continued training ⇒ forced relocation into free heads.
  control : no lesion, continue training ⇒ does wikitext FT itself move induction? (baseline)

Measures over continued-training steps: ICL drop (functional), max induction score among NON-clamped heads
(relocation target), recovery step (ICL back to 50% of pre-lesion baseline).

  python pythia_regen.py --model EleutherAI/pythia-70m --step 143000 --train_steps 300 --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pythia_redundancy as PR          # reuse: get_layers_and_proj, install_hooks, ABLATE, induction_per_head, icl_drop, make_batch

RESULTS = PR.RESULTS


def load_wikitext_ids(tok, n_tokens=400_000):
    import datasets
    d = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    buf, tot = [], 0
    for row in d:
        t = row["text"]
        if not t or not t.strip():
            continue
        ids = tok(t).input_ids
        buf.extend(ids); tot += len(ids)
        if tot >= n_tokens:
            break
    return np.array(buf[:n_tokens], dtype=np.int64)


def train_batch(ids, B, T, rng, device):
    starts = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[s:s + T] for s in starts])
    return torch.from_numpy(x).to(device)


def induction_train_batch(vocab, B, T, rng, device):
    """PURE induction pressure: fresh random tokens tiled twice each batch (max ΔL* for induction).
    Fresh per batch ⇒ can't memorize, must (re)learn the induction ALGORITHM."""
    half = rng.integers(0, vocab, size=(B, T // 2), dtype=np.int64)
    return torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)


@torch.no_grad()
def induction_free(model, probe, L, Sset, nh):
    """max induction score among heads NOT in the clamped set S (the relocation readout)."""
    scores = PR.induction_per_head(model, probe, L)
    free = [(lh, v) for lh, v in scores.items() if lh not in Sset]
    free.sort(key=lambda kv: kv[1], reverse=True)
    return free[0]                       # ((l,h), score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--step", type=int, default=143000)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)   # re-warm to drive re-formation
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--probe_batch", type=int, default=8)
    ap.add_argument("--train_batch", type=int, default=8)
    ap.add_argument("--train_seq", type=int, default=256)
    ap.add_argument("--max_clamp", type=int, default=15)
    ap.add_argument("--data", choices=["induction", "wikitext", "localtext", "mix"], default="induction",
                    help="induction=pure repeat; wikitext=HF natural; localtext=--textfile; mix=half induction half localtext")
    ap.add_argument("--textfile", default=None, help="local natural-text file for --data localtext/mix")
    ap.add_argument("--control", action="store_true", help="no lesion baseline")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_local = os.path.isdir(args.model)                    # local flat dir -> no revision (weights already a fixed step)
    tok = AutoTokenizer.from_pretrained(args.model)
    probe = PR.make_batch(tok.vocab_size, args.L, args.probe_batch, device, seed=0)
    kw = dict(dtype=torch.float32, attn_implementation="eager")
    if not is_local:
        kw["revision"] = f"step{args.step}"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).to(device)
    attns, proj_name, head_size, nh = PR.get_layers_and_proj(model)
    nl = len(attns)
    print(f"device={device} model={args.model} step={args.step} {nl}L×{nh}H lr={args.lr} train_steps={args.train_steps}")

    # baseline + build induction SET S (greedy until ICL<=20% base)
    model.eval()
    base_icl = PR.icl_drop(model, probe, args.L)
    scores = PR.induction_per_head(model, probe, args.L)
    ranked = sorted(scores, key=lambda lh: scores[lh], reverse=True)
    PR.ABLATE.clear(); PR.install_hooks(attns, proj_name, head_size)
    S = []
    for k in range(1, args.max_clamp + 1):
        l, h = ranked[k - 1]; PR.ABLATE.setdefault(l, set()).add(h); S.append((l, h))
        if PR.icl_drop(model, probe, args.L) <= 0.20 * base_icl:
            break
    Sset = set(S)
    crater_icl = PR.icl_drop(model, probe, args.L)
    print(f"  baseline ICL={base_icl:.2f} nats; induction SET |S|={len(S)} (heads {S[:6]}{'...' if len(S)>6 else ''})")
    print(f"  after CLAMP: ICL={crater_icl:.2f} ({crater_icl/base_icl*100:.0f}% of base)  [induction destroyed]")

    if args.control:                     # control = remove the lesion, just continue training
        PR.ABLATE.clear()
        print("  [control mode] no lesion maintained")

    # continue training, lesion (S) maintained via the persistent hooks
    if args.data == "wikitext":
        ids = load_wikitext_ids(tok, n_tokens=200_000 if args.smoke else 600_000)
    elif args.data in ("localtext", "mix"):
        txt = open(args.textfile, encoding="utf-8").read()
        ids = np.array(tok(txt).input_ids, dtype=np.int64); print(f"  localtext {len(ids)} tokens from {args.textfile}")
    else:
        ids = None
    rng = np.random.default_rng(0)
    def next_batch():
        if args.data in ("wikitext", "localtext"):
            return train_batch(ids, args.train_batch, args.train_seq, rng, device)
        if args.data == "mix":                              # half natural text, half induction pressure
            if rng.random() < 0.5:
                return train_batch(ids, args.train_batch, args.train_seq, rng, device)
            return induction_train_batch(tok.vocab_size, args.train_batch, args.train_seq, rng, device)
        return induction_train_batch(tok.vocab_size, args.train_batch, args.train_seq, rng, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    curve = []
    def snap(step):
        model.eval()
        icl = PR.icl_drop(model, probe, args.L)
        (lh, sc) = induction_free(model, probe, args.L, Sset, nh)
        curve.append(dict(step=step, icl=icl, free_top_head=list(lh), free_top_score=sc))
        print(f"  contd-step {step:>4}: ICL={icl:.2f} ({icl/base_icl*100:.0f}% base)  "
              f"top free head L{lh[0]}H{lh[1]} score={sc:.2f}")
        model.train()
    snap(0)
    for step in range(1, args.train_steps + 1):
        model.train()
        x = next_batch()
        out = model(x, labels=x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % args.eval_every == 0:
            snap(step)

    # report
    icls = np.array([c["icl"] for c in curve]); steps = np.array([c["step"] for c in curve])
    half = 0.5 * base_icl
    rec = next((int(s) for s, i in zip(steps, icls) if i >= half), None)
    final = icls[-1]
    print(f"\n=== REGEN-Pythia ({'CONTROL' if args.control else 'CLAMP'}) ===")
    print(f"  pre-lesion ICL {base_icl:.2f} -> crater {crater_icl:.2f} -> after {args.train_steps} contd-steps {final:.2f} "
          f"({final/base_icl*100:.0f}% recovered)")
    print(f"  recovery to 50% base at contd-step = {rec if rec is not None else 'NOT within budget'}")
    fh = curve[-1]["free_top_head"]
    print(f"  induction now carried by free head L{fh[0]}H{fh[1]} (clamped set S frozen) "
          f"⇒ {'RELOCATED into surviving heads' if final > 0.4*base_icl else 'did NOT regrow'}")
    res = dict(model=args.model, step=args.step, mode="control" if args.control else "clamp", data=args.data,
               lr=args.lr, base_icl=base_icl, crater_icl=crater_icl, nS=len(S), S=[list(x) for x in S],
               recovery_step=rec, final_icl=float(final), curve=curve)
    tagm = ("control" if args.control else "clamp") + "_" + args.data
    json.dump(res, open(os.path.join(RESULTS, f"pythia_regen_{args.model.split('/')[-1]}_{tagm}.json"), "w"), indent=2)
    print(f"  saved results/pythia_regen_{args.model.split('/')[-1]}_{tagm}.json")


if __name__ == "__main__":
    main()
