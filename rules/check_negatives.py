"""
check_negatives.py - diagnostic: do certain protein families in the negative
set explain the weak separation?

Usage:
    python check_negatives.py --eval eval_set.csv --negatives neg_clean.csv

It compares three groups on net charge, hydrophobic fraction, hydrophobic
moment and length:
    AMP          - positives in eval_set.csv
    neg_flagged  - negatives whose UniProt name matches the pattern
    neg_other    - all other negatives
and writes neg_filtered.csv (negatives without the flagged families).

This is a DIAGNOSTIC. Removing "hard" negatives makes any result look better,
so if you report numbers from the filtered set, report them next to the
unfiltered ones and say how the filter was chosen.
"""
import argparse

import pandas as pd

PATTERN = "ribosomal|histone|transmembrane|cytochrome|NADH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_set.csv")
    ap.add_argument("--negatives", default="neg_clean.csv")
    ap.add_argument("--pattern", default=PATTERN)
    ap.add_argument("--out", default="neg_filtered.csv")
    args = ap.parse_args()

    ev = pd.read_csv(args.eval)
    neg = pd.read_csv(args.negatives)
    flagged = neg["header"].str.contains(args.pattern, case=False, na=False)

    print(f"Negatives flagged by pattern: {flagged.sum()} of {len(neg)} "
          f"({flagged.mean():.1%})")

    ev = ev.merge(neg[["sequence", "header"]], on="sequence", how="left")
    ev["group"] = ev["label"].map({1: "AMP", 0: "neg_other"})
    is_flag = (ev["label"] == 0) & ev["header"].str.contains(args.pattern, case=False, na=False)
    ev.loc[is_flag, "group"] = "neg_flagged"

    cols = ["NetCharge", "HydrophobicFraction", "HydrophobicMoment", "Length"]
    summary = ev.groupby("group")[cols].median().round(2)
    summary.insert(0, "n", ev.groupby("group").size())
    print("\nMedian values per group:")
    print(summary.to_string())

    neg[~flagged].to_csv(args.out, index=False)
    print(f"\nSaved {(~flagged).sum()} negatives without the flagged families to {args.out}")
    print("To re-run the tuning on them:")
    print(f"  python tune_rules.py --positives amp_clean.csv --negatives {args.out} --out eval_filtered.csv")


if __name__ == "__main__":
    main()