"""
eval_similarity.py - how well does "has a close known homolog" separate AMPs
from non-AMPs?

Usage:
    python eval_similarity.py --eval eval_set.csv --db amp.db --n 300

Method
------
- Queries are a random sample of n positives and n negatives from the balanced,
  length-matched eval_set.csv written by tune_rules.py.
- Each query is searched against amp.db with similarity.find_similar.
- For a positive query, its own entry in the database is skipped (leave-self-out),
  so the best remaining hit is the nearest OTHER known AMP.
- A query "fires" if its best hit has identity >= t and coverage >= --min-coverage.
- Because DRAMP contains many near-identical variants (melittin analogs, etc.),
  positives look easier than a genuinely new peptide would. So sensitivity is
  reported twice: for all positives, and for the "novel" subset whose nearest
  other AMP is below --novel-below % identity.

Every query result is saved to similarity_eval.csv, so you can re-summarize
without re-running the alignments:
    python eval_similarity.py --reuse similarity_eval.csv

Speed: each query is aligned against ~10,000 peptides. Try --n 30 first and
see how long it takes before running a larger sample.
"""
import argparse
import time

import numpy as np
import pandas as pd

THRESHOLDS = [40, 50, 60, 70, 80, 90]


def run_queries(df, reference):
    from similarity import find_similar

    rows, t0 = [], time.time()
    for i, (seq, label) in enumerate(zip(df["sequence"], df["label"]), 1):
        hits = find_similar(seq, reference, top_n=2)
        hit = next((h for h in hits if h["sequence"] != seq), None)
        rows.append(
            {
                "sequence": seq,
                "label": int(label),
                "best_id": hit["peptide_id"] if hit else None,
                "identity_pct": hit["identity_pct"] if hit else 0.0,
                "coverage_pct": hit["coverage_pct"] if hit else 0.0,
                "score_ratio": hit["score_ratio"] if hit else 0.0,
            }
        )
        if i % 50 == 0 or i == len(df):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(df) - i)
            print(f"  {i}/{len(df)} queries done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")
    return pd.DataFrame(rows)


def summarize(res, min_coverage, novel_below):
    pos = res[res["label"] == 1]
    neg = res[res["label"] == 0]
    novel = pos[pos["identity_pct"] < novel_below]

    near_dup = ((pos["identity_pct"] >= 90) & (pos["coverage_pct"] >= min_coverage)).mean()
    print(f"\nQueries: {len(pos)} positives, {len(neg)} negatives")
    print(f"Positives whose nearest OTHER known AMP is >=90% identical: {near_dup:.1%}")
    print(f"'Novel' positives (nearest other AMP < {novel_below}% identity): "
          f"{len(novel)} ({len(novel) / len(pos):.1%})")
    print(f"Coverage required: >= {min_coverage}%\n")

    def fire(d, t):
        return (d["identity_pct"] >= t) & (d["coverage_pct"] >= min_coverage)

    rows = []
    for t in THRESHOLDS:
        tp, fp = int(fire(pos, t).sum()), int(fire(neg, t).sum())
        rows.append(
            {
                "identity >=": t,
                "sens (all pos)": round(tp / len(pos), 3),
                "sens (novel pos)": round(float(fire(novel, t).mean()), 3) if len(novel) else np.nan,
                "false pos rate (neg)": round(fp / len(neg), 3),
                "precision (all)": round(tp / (tp + fp), 3) if tp + fp else np.nan,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n(amp_rules.py currently uses identity >= 70 and coverage >= 70.)")
    print("Read 'sens (novel pos)' as the realistic figure for a new peptide;")
    print("'sens (all pos)' is inflated by near-duplicate variants in the database.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_set.csv")
    ap.add_argument("--db", default="amp.db")
    ap.add_argument("--n", type=int, default=300, help="queries per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-coverage", type=float, default=70.0)
    ap.add_argument("--novel-below", type=float, default=90.0)
    ap.add_argument("--out", default="similarity_eval.csv")
    ap.add_argument("--reuse", default=None, help="summarize an existing results CSV")
    args = ap.parse_args()

    if args.reuse:
        res = pd.read_csv(args.reuse)
    else:
        from similarity import load_reference

        ev = pd.read_csv(args.eval)
        sample = pd.concat(
            [
                ev[ev["label"] == lab].sample(min(args.n, int((ev["label"] == lab).sum())),
                                              random_state=args.seed)
                for lab in (1, 0)
            ]
        ).reset_index(drop=True)
        reference = load_reference(args.db)
        print(f"Searching {len(sample)} queries against {len(reference)} reference peptides...")
        res = run_queries(sample, reference)
        res.to_csv(args.out, index=False)
        print(f"Saved per-query results to {args.out}")

    summarize(res, args.min_coverage, args.novel_below)


if __name__ == "__main__":
    main()