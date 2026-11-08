import argparse
 
import numpy as np
import pandas as pd
 
from amp_rules import assess
from features import extract_features
 
FEATURES = [
    "Length", "NetCharge", "pI", "AvgHydrophobicity", "HydrophobicMoment",
    "HydrophobicFraction", "CationicRatio", "AnionicRatio",
    "AliphaticRatio", "AromaticRatio",
]
 
 
def balance(pos, neg, bin_width, seed):
    pos = pos.assign(bin=pos["length"] // bin_width)
    neg = neg.assign(bin=neg["length"] // bin_width)
    keep_p, keep_n = [], []
    for b in sorted(set(pos["bin"]) & set(neg["bin"])):
        p, n = pos[pos["bin"] == b], neg[neg["bin"] == b]
        k = min(len(p), len(n))
        keep_p.append(p.sample(k, random_state=seed))
        keep_n.append(n.sample(k, random_state=seed))
    return pd.concat(keep_p), pd.concat(keep_n)
 
 
def auc(values, labels):
    """Rank-based AUC (probability a random positive scores higher than a random negative)."""
    ranks = pd.Series(values).rank().to_numpy()
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
 
 
def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
    return {"sensitivity": sens, "specificity": spec, "precision": prec,
            "F1": f1, "balanced_acc": (sens + spec) / 2,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}
 
 
def fmt(m):
    return (f"sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}  "
            f"prec {m['precision']:.3f}  F1 {m['F1']:.3f}  "
            f"bal.acc {m['balanced_acc']:.3f}  "
            f"(TP {m['TP']} FP {m['FP']} TN {m['TN']} FN {m['FN']})")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", default="amp_clean.csv")
    ap.add_argument("--negatives", default="neg_clean.csv")
    ap.add_argument("--bin", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_set.csv", help="saves the balanced set with features")
    args = ap.parse_args()
 
    pos = pd.read_csv(args.positives)
    neg = pd.read_csv(args.negatives)
    pos_b, neg_b = balance(pos, neg, args.bin, args.seed)
    print(f"Positives available {len(pos)}, negatives available {len(neg)}")
    print(f"Balanced set: {len(pos_b)} positives + {len(neg_b)} negatives "
          f"(dropped {len(pos) - len(pos_b)} positives, mostly in length bins without enough negatives)")
    print(f"Length range kept: {int(min(pos_b.length.min(), neg_b.length.min()))}-"
          f"{int(max(pos_b.length.max(), neg_b.length.max()))} aa\n")
 
    df = pd.concat([pos_b.assign(label=1), neg_b.assign(label=0)], ignore_index=True)
    df = df[["sequence", "length", "label"]]
 
    print("Computing features...")
    feats = pd.DataFrame([extract_features(s) for s in df["sequence"]])
    df = pd.concat([df.drop(columns=["length"]), feats[FEATURES]], axis=1)
    df.to_csv(args.out, index=False)
 
    # Stratified 50/50 split
    rng = np.random.default_rng(args.seed)
    df["half"] = ""
    for lab in (0, 1):
        idx = df.index[df["label"] == lab].to_numpy().copy()
        rng.shuffle(idx)
        df.loc[idx[: len(idx) // 2], "half"] = "tune"
        df.loc[idx[len(idx) // 2:], "half"] = "test"
    tune, test = df[df["half"] == "tune"], df[df["half"] == "test"]
    print(f"Tuning half: {len(tune)}   Test half: {len(test)}\n")
 
    # ---- Per-feature table
    print("PER-FEATURE SEPARATION (AUC on all balanced data; cutoff chosen on tuning half)")
    rows = []
    for f in FEATURES:
        a = auc(df[f].to_numpy(), df["label"].to_numpy())
        direction = ">=" if a >= 0.5 else "<="
        cands = np.unique(np.quantile(tune[f], np.linspace(0.02, 0.98, 97)))
        best, best_j = None, -2
        for c in cands:
            pred = (tune[f] >= c) if direction == ">=" else (tune[f] <= c)
            m = metrics(tune["label"], pred)
            j = m["sensitivity"] + m["specificity"] - 1
            if j > best_j:
                best, best_j = c, j
        pred_t = (test[f] >= best) if direction == ">=" else (test[f] <= best)
        mt = metrics(test["label"], pred_t)
        rows.append({"feature": f, "AUC": round(a, 3), "rule": f"{direction} {best:.3f}",
                     "test_sens": round(mt["sensitivity"], 3),
                     "test_spec": round(mt["specificity"], 3)})
    print(pd.DataFrame(rows).sort_values("AUC", ascending=False, key=lambda s: (s - 0.5).abs())
          .to_string(index=False))
    print("(Length should be near 0.5 if the length matching worked.)\n")
 
    # ---- Rule combination search (tuning half only)
    print("RULE SEARCH on tuning half (criteria met >= k), evaluated on test half")
    def arr(d):
        return (d["NetCharge"].to_numpy(), d["HydrophobicFraction"].to_numpy(),
                d["HydrophobicMoment"].to_numpy(), d["Length"].to_numpy(),
                d["label"].to_numpy())
    ch_t, fr_t, mo_t, le_t, y_t = arr(tune)
    ch_e, fr_e, mo_e, le_e, y_e = arr(test)
 
    def predict(ch, fr, mo, le, p):
        crit = [(ch >= p["charge"]), (fr >= p["h_lo"]) & (fr <= p["h_hi"]),
                (le >= p["len_lo"]) & (le <= p["len_hi"])]
        if p["moment"] is not None:
            crit.append(mo >= p["moment"])
        return np.sum(crit, axis=0) >= p["k"], len(crit)
 
    best_p, best_score = None, -1
    for charge in [0, 1, 1.5, 2, 3, 4, 5]:
        for h_lo in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            for h_hi in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
                for moment in [None, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                    for len_lo, len_hi in [(10, 50), (10, 60), (10, 100)]:
                        n_crit = 4 if moment is not None else 3
                        for k in range(1, n_crit + 1):
                            p = dict(charge=charge, h_lo=h_lo, h_hi=h_hi, moment=moment,
                                     len_lo=len_lo, len_hi=len_hi, k=k)
                            pred, _ = predict(ch_t, fr_t, mo_t, le_t, p)
                            m = metrics(y_t, pred)
                            if m["balanced_acc"] > best_score:
                                best_score, best_p = m["balanced_acc"], p
    print("Best combination found on tuning half:")
    print("  ", best_p)
    pred_e, n_crit = predict(ch_e, fr_e, mo_e, le_e, best_p)
    print(f"  tuning half (optimistic): bal.acc {best_score:.3f}")
    print(f"  TEST half: {fmt(metrics(y_e, pred_e))}")
    print(f"  (rule = at least {best_p['k']} of {n_crit} criteria met)\n")
 
    # ---- Current rules in amp_rules.py
    print("CURRENT amp_rules.py (properties only, 'Possible AMP' counted as positive), TEST half")
    cur = [assess(s)["verdict"] == "Possible AMP" for s in test["sequence"]]
    print("  ", fmt(metrics(y_e, np.array(cur))))
    print("\nSaved balanced feature table to", args.out)
 
 
if __name__ == "__main__":
    main()