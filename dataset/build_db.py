import argparse, datetime, os, sqlite3, pandas as pd

SCHEMA = """
DROP TABLE IF EXISTS amp_sequences;
DROP TABLE IF EXISTS db_info;

CREATE TABLE amp_sequences (
    peptide_id TEXT PRIMARY KEY,
    header TEXT,
    sequence TEXT NOT NULL UNIQUE,
    length INTEGER NOT NULL,
    activity TEXT,
    source TEXT,
    reference TEXT
);
CREATE INDEX idx_amp_length ON amp_sequences(length);

CREATE TABLE db_info (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="cleaned CSV from clean_amp.py")
    ap.add_argument("--db", default="amp.db")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"id", "header", "sequence", "length", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {', '.join(sorted(missing))}")

    for col in ("source", "reference"):
        if col not in df.columns:
            df[col] = None

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)

    rows = df[
        ["id", "header", "sequence", "length", "activity", "source", "reference"]
    ].values.tolist()

    con.executemany(
        "INSERT INTO amp_sequences VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )

    info = {
        "source_file": os.path.basename(args.csv),
        "n_sequences": str(len(df)),
        "built_on": datetime.date.today().isoformat(),
        "dataset": "DRAMP general AMPs, cleaned (standard residues, length filtered, deduplicated)",
    }
    con.executemany("INSERT INTO db_info VALUES (?, ?)", list(info.items()))
    con.commit()

    n = con.execute("SELECT COUNT(*) FROM amp_sequences").fetchone()[0]
    lo, hi = con.execute("SELECT MIN(length), MAX(length) FROM amp_sequences").fetchone()
    print(f"Built {args.db}: {n} sequences, length {lo}-{hi}")
    print("Example rows:")
    for r in con.execute("SELECT peptide_id, header, length FROM amp_sequences LIMIT 3"):
        print("  ", r)
    con.close()

if __name__ == "__main__":
    main()