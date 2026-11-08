import argparse, random, re
from collections import defaultdict
import pandas as pd

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
 
 
def read_fasta(path):
    header, seq = None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line[1:].strip(), []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="UniProt FASTA file(s)")
    ap.add_argument("--positives", default="amp_clean.csv")
    ap.add_argument("--out", default="neg_clean.csv")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--bin", type=int, default=5, help="length bin width for matching")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-match", action="store_true", help="skip length matching")
    args = ap.parse_args()
 
    pos = pd.read_csv(args.positives)
    pos_seqs = set(pos["sequence"])
 
    rows = []
    for path in args.files:
        for header, seq in read_fasta(path):
            rows.append({"header": header, "sequence": seq})
    df = pd.DataFrame(rows)
    stats = {"raw entries": len(df)}
 
    df["sequence"] = df["sequence"].str.upper().str.replace(r"[\s\d\-\*\.]", "", regex=True)
    df = df[df["sequence"].str.len() > 0]
 
    ok = df["sequence"].apply(lambda s: set(s) <= STANDARD_AA)
    stats["removed: non-standard residues"] = int((~ok).sum())
    df = df[ok]
 
    df["length"] = df["sequence"].str.len()
    in_range = df["length"].between(args.min_len, args.max_len)
    stats[f"removed: length outside {args.min_len}-{args.max_len}"] = int((~in_range).sum())
    df = df[in_range]
 
    before = len(df)
    df = df.drop_duplicates(subset="sequence")
    stats["removed: duplicate sequences"] = before - len(df)
 
    is_pos = df["sequence"].isin(pos_seqs)
    stats["removed: identical to a positive"] = int(is_pos.sum())
    df = df[~is_pos]
    stats["candidate negatives"] = len(df)
 
    if not args.no_match:
        rng = random.Random(args.seed)
        pos_bins = pos["length"].apply(lambda n: n // args.bin).value_counts().to_dict()
        by_bin = defaultdict(list)
        for idx, n in zip(df.index, df["length"]):
            by_bin[n // args.bin].append(idx)
 
        keep, short_bins = [], []
        for b, want in sorted(pos_bins.items()):
            have = by_bin.get(b, [])
            take = min(want, len(have))
            keep.extend(rng.sample(have, take))
            if take < want:
                short_bins.append((b * args.bin, b * args.bin + args.bin - 1, want, take))
        df = df.loc[keep]
        stats["negatives after length matching"] = len(df)
    else:
        short_bins = []
 
    df = df.sort_values("length").reset_index(drop=True)
    df.insert(0, "id", [f"NEG{i:05d}" for i in range(1, len(df) + 1)])
    df["source"] = "UniProtKB/Swiss-Prot"
    df["label"] = 0
    df = df[["id", "header", "sequence", "length", "source", "label"]]
    df.to_csv(args.out, index=False)
 
    print("Negative set report")
    print("-------------------")
    for k, v in stats.items():
        print(f"{k:45s} {v}")
    if short_bins:
        print("\nLength bins with fewer negatives than positives (lo-hi: wanted, got):")
        for lo, hi, want, got in short_bins:
            print(f"  {lo}-{hi}: {want}, {got}")
        print("The classes are not balanced in these bins. Report this, or")
        print("down-sample the positives in these bins when evaluating.")
    print(f"\nSaved {len(df)} negatives to {args.out} (positives: {len(pos)})")
 
 
if __name__ == "__main__":
    main()