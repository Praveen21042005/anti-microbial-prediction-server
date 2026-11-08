import argparse, sqlite3
from Bio import Align
from Bio.Align import substitution_matrices

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def make_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner

def load_reference(db_path="amp.db"):
    """Return a list of (peptide_id, header, sequence, activity) from the database."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT peptide_id, header, sequence, activity FROM amp_sequences"
    ).fetchall()
    con.close()
    return rows

def _alignment_stats(aln, query, target):
    """Percent identity over the aligned region and percent of the query covered."""
    q_blocks, t_blocks = aln.aligned
    matches = columns = aligned_query = 0
    prev_q_end = prev_t_end = None
    for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
        length = qe - qs
        matches += sum(1 for a, b in zip(query[qs:qe], target[ts:te]) if a == b)
        columns += length
        aligned_query += length
        if prev_q_end is not None:
            # gap columns between this block and the previous one
            columns += (qs - prev_q_end) + (ts - prev_t_end)
        prev_q_end, prev_t_end = qe, te
    identity = 100.0 * matches / columns if columns else 0.0
    coverage = 100.0 * aligned_query / len(query)
    return identity, coverage

def find_similar(query, reference, top_n=5):
    """
    Return the top_n most similar reference peptides as a list of dicts with:
    peptide_id, header, activity, sequence, score, score_ratio,
    identity_pct, coverage_pct.
 
    score_ratio = alignment score / the query's score against itself (1.0 = identical).
    """
    query = query.upper().strip().replace(" ", "")
    if not query or not set(query) <= STANDARD_AA:
        raise ValueError("Query must contain only the 20 standard amino acids.")
 
    aligner = make_aligner()
    self_score = aligner.score(query, query)
 
    # Fast pass: score only, no alignment objects
    scored = [(aligner.score(query, seq), i) for i, (_, _, seq, _) in enumerate(reference)]
    scored.sort(reverse=True)
 
    hits = []
    for score, i in scored[:top_n]:
        pid, header, seq, activity = reference[i]
        aln = next(iter(aligner.align(query, seq)))
        identity, coverage = _alignment_stats(aln, query, seq)
        hits.append(
            {
                "peptide_id": pid,
                "header": header,
                "activity": activity,
                "sequence": seq,
                "score": round(float(score), 1),
                "score_ratio": round(float(score) / self_score, 3) if self_score else 0.0,
                "identity_pct": round(identity, 1),
                "coverage_pct": round(coverage, 1),
            }
        )
    return hits
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("--db", default="amp.db")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
 
    reference = load_reference(args.db)
    print(f"Searching {len(reference)} reference peptides...\n")
    for rank, h in enumerate(find_similar(args.sequence, reference, args.top), 1):
        print(
            f"{rank}. {h['peptide_id']}  identity {h['identity_pct']}%  "
            f"coverage {h['coverage_pct']}%  score {h['score']}"
        )
        print(f"   {h['header']}")
        print(f"   {h['sequence']}\n")
 
 
if __name__ == "__main__":
    main()
