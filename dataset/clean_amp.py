import argparse, os, re, pandas as pd

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def read_fasta(path):
    """Yield (header, sequence) from a FASTA file. Handles multi-line sequences."""
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
    ap.add_argument("files",nargs="+",help="FASTA files to merge")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--out", default="amp_clean.csv")
    args = ap.parse_args()

    rows = []
    for path in args.files:
        activity = os.path.splitext(os.path.basename(path))[0]
        for header, seq in read_fasta(path):
            rows.append({"header": header, "sequence": seq, "activity": activity})

    df = pd.DataFrame(rows)
    stats = {"raw entries" : len(df)}

    #cleaning data using regex
    df["sequence"] = (
        df["sequence"].str.upper().str.replace(r"[\s\d\-\*\.]", "", regex=True)
    )

    #dropping empty sequences
    df = df[df["sequence"].str.len() > 0]
    stats["after removing empty"] = len(df)

    #dropping sequences with non-standard residues
    ok = df["sequence"].apply(lambda s: set(s) <= STANDARD_AA)
    stats["removed: non-standard residues"] = int((~ok).sum())
    df = df[ok]

    #length filter
    df["length"] = df["sequence"].str.len()
    in_range = df["length"].between(args.min_len, args.max_len)
    stats[f"removed: length outside {args.min_len} - {args.max_len}"] = int((~in_range).sum())
    df = df[in_range]

    #removing exact duplicate sequences, merging their activites and headers
    df = (
        df.groupby("sequence", as_index=False).agg(
            header=("header","first"),
            activity=("activity",lambda x: ";".join(sorted(set(x)))),
        )
    )
    df["length"] = df["sequence"].str.len()
    stats["unique sequences kept"] = len(df)

    #label and id
    df["label"] = 1
    df.insert(0, "id", [f"AMP{i:05d}" for i in range(1, len(df) + 1)])
    df.to_csv(args.out,index=False)
    print("Cleaning data")

    for k, v in stats.items():
        print(f"{k:45s} {v}")
    print(f"\nSaved {len(df)} sequences to {args.out}")

if __name__ == "__main__":
    main()


