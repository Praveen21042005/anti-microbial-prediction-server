"""
app_helpers.py - everything the web app does that is not screen layout.

No Streamlit import here, so these functions can be tested on their own.
"""
import datetime
import sqlite3

import pandas as pd

from amp_rules import assess
from features import (
    AMINO_ACIDS,
    calculate_hydrophobic_moment,
    calculate_net_charge,
    validate_sequence,
)

MAX_BATCH = 200          # sequences per batch run
MIN_LEN, MAX_LEN = 5, 100


# ---------------------------------------------------------------- input parsing
def parse_fasta(text):
    """Return a list of (header, sequence). Sequences without a header get Seq_N."""
    entries, header, seq = [], None, []

    def flush():
        nonlocal header, seq
        if seq:
            entries.append((header or f"Seq_{len(entries) + 1}", "".join("".join(seq).split())))
        header, seq = None, []

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            parts = line[1:].split()
            header = parts[0] if parts else None
        else:
            seq.append(line)
    flush()
    return entries


def sequence_from_input(text):
    """Accept a raw sequence or a FASTA record (first record only)."""
    text = text.strip()
    if text.startswith(">"):
        entries = parse_fasta(text)
        return entries[0][1] if entries else ""
    return "".join(text.split())


# -------------------------------------------------------------------- analysis
def analyse(seq, reference, top_n=5):
    """Similarity search + rule-based verdict for one validated sequence."""
    from similarity import find_similar

    hits = find_similar(seq, reference, top_n=top_n)
    return assess(seq, hits), hits


def criteria_table(result):
    rows = []
    for c in result["criteria"]:
        rows.append(
            {
                "Criterion": c["name"],
                "Value": c["value"],
                "Rule": c["rule"],
                "Met": "yes" if c["met"] else "no",
                "Used in verdict": "yes" if c["scored"] else "information only",
            }
        )
    best = result["best_hit"]
    if best:
        rows.append(
            {
                "Criterion": "Close known relative",
                "Value": f"{best['identity_pct']}% identity, {best['coverage_pct']}% coverage",
                "Rule": ">= 70% identity and >= 70% coverage",
                "Met": "yes" if result["homolog_found"] else "no",
                "Used in verdict": "yes",
            }
        )
    return pd.DataFrame(rows)


def hits_table(hits):
    return pd.DataFrame(
        [
            {
                "Rank": i,
                "Peptide ID": h["peptide_id"],
                "Identity (%)": h["identity_pct"],
                "Coverage (%)": h["coverage_pct"],
                "Sequence": h["sequence"],
                "Name / header": h["header"],
                "Activity": h["activity"],
            }
            for i, h in enumerate(hits, 1)
        ]
    )


def properties_table(feats):
    return pd.DataFrame(
        {
            "Property": [
                "Sequence length", "Molecular weight", "Isoelectric point (pI)",
                "Net charge (pH 7.4)", "Mean hydrophobicity", "Hydrophobic moment (uH)",
                "Hydrophobic residues", "Cationic residues (K, R, H)",
                "Anionic residues (D, E)", "Aromatic residues (F, W, Y)",
            ],
            "Value": [
                f"{feats['Length']} aa", f"{feats['MW_Da']} Da", f"{feats['pI']}",
                f"{feats['NetCharge']:+.2f}", f"{feats['AvgHydrophobicity']:.3f}",
                f"{feats['HydrophobicMoment']:.3f}",
                f"{feats['HydrophobicFraction']:.1%}", f"{feats['CationicRatio']:.1%}",
                f"{feats['AnionicRatio']:.1%}", f"{feats['AromaticRatio']:.1%}",
            ],
        }
    )


def composition_table(feats):
    return pd.DataFrame(
        {"Amino acid": list(AMINO_ACIDS),
         "Frequency (%)": [feats[f"AAC_{aa}"] for aa in AMINO_ACIDS]}
    )


def build_report(seq, result, hits, info=None):
    """Plain-text report for download. Nothing is stored on the server."""
    f = result["features"]
    lines = [
        "AMP-Predict report",
        f"Date: {datetime.date.today().isoformat()}",
        "",
        f"Sequence ({f['Length']} aa): {seq}",
        f"Verdict: {result['verdict']}",
        "",
        "Criteria",
    ]
    for _, r in criteria_table(result).iterrows():
        lines.append(f"  - {r['Criterion']}: {r['Value']} | rule: {r['Rule']} | "
                     f"met: {r['Met']} | {r['Used in verdict']}")
    lines += ["", "Properties"]
    for _, r in properties_table(f).iterrows():
        lines.append(f"  - {r['Property']}: {r['Value']}")
    lines += ["", "Closest known peptides"]
    for _, r in hits_table(hits).iterrows():
        lines.append(f"  {r['Rank']}. {r['Peptide ID']}  identity {r['Identity (%)']}%  "
                     f"coverage {r['Coverage (%)']}%  {r['Sequence']}")
    if info:
        lines += ["", f"Reference database: {info.get('n_sequences', '?')} known AMPs "
                      f"({info.get('dataset', 'DRAMP')}, built {info.get('built_on', '?')})"]
    lines += [
        "",
        "How to read this: the verdict is a rule-based indicator, not a probability and not",
        "proof of activity. 'Unlikely AMP' means the sequence does not look like a typical",
        "AMP; it does not show that the peptide is inactive.",
    ]
    return "\n".join(lines)


def batch_assess(entries, reference, progress=None):
    """Run the analysis on many (header, sequence) pairs. Returns a DataFrame."""
    rows = []
    total = len(entries)
    for i, (header, raw) in enumerate(entries, 1):
        ok, seq_or_err = validate_sequence(raw, MIN_LEN, MAX_LEN)
        if not ok:
            rows.append(
                {"Header": header, "Sequence": raw, "Length": len(raw),
                 "Verdict": "Invalid input", "Net charge": None,
                 "Hydrophobic moment": None, "Closest known peptide": None,
                 "Identity (%)": None, "Coverage (%)": None, "Note": seq_or_err}
            )
        else:
            result, hits = analyse(seq_or_err, reference, top_n=1)
            f, best = result["features"], result["best_hit"]
            rows.append(
                {"Header": header, "Sequence": seq_or_err, "Length": f["Length"],
                 "Verdict": result["verdict"], "Net charge": f["NetCharge"],
                 "Hydrophobic moment": f["HydrophobicMoment"],
                 "Closest known peptide": best["peptide_id"] if best else None,
                 "Identity (%)": best["identity_pct"] if best else None,
                 "Coverage (%)": best["coverage_pct"] if best else None,
                 "Note": ""}
            )
        if progress:
            progress(i, total)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- database
def _like(q):
    return "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def search_database(db_path, query="", length_range=None, limit=500):
    """Returns (DataFrame of up to `limit` rows, total number of matches)."""
    where, params = [], []
    if query.strip():
        p = _like(query.strip())
        cols = ["peptide_id", "header", "sequence",
                "COALESCE(activity, '')", "COALESCE(source, '')"]
        where.append("(" + " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for c in cols) + ")")
        params += [p] * len(cols)
    if length_range:
        where.append("length BETWEEN ? AND ?")
        params += [int(length_range[0]), int(length_range[1])]
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    con = sqlite3.connect(db_path)
    try:
        total = con.execute(f"SELECT COUNT(*) FROM amp_sequences{clause}", params).fetchone()[0]
        df = pd.read_sql_query(
            "SELECT peptide_id AS 'Peptide ID', header AS 'Name / header', "
            "sequence AS 'Sequence', length AS 'Length', activity AS 'Activity', "
            f"source AS 'Source' FROM amp_sequences{clause} ORDER BY peptide_id LIMIT ?",
            con, params=params + [int(limit)],
        )
    finally:
        con.close()
    return df, total


def db_info(db_path):
    """Info about the reference database (counts, build date, length range)."""
    con = sqlite3.connect(db_path)
    try:
        info = dict(con.execute("SELECT key, value FROM db_info").fetchall())
        n, lo, hi = con.execute(
            "SELECT COUNT(*), MIN(length), MAX(length) FROM amp_sequences"
        ).fetchone()
    finally:
        con.close()
    info.update({"count": n, "min_len": lo, "max_len": hi})
    return info


def db_property_sample(db_path, n=2000, seed=42):
    """Net charge and hydrophobic moment for a random sample of the database."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT peptide_id, sequence, length FROM amp_sequences", con
        )
    finally:
        con.close()
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    df = df.assign(
        NetCharge=[round(calculate_net_charge(s), 2) for s in df["sequence"]],
        HydrophobicMoment=[round(calculate_hydrophobic_moment(s), 3) for s in df["sequence"]],
    )
    return df.reset_index(drop=True)