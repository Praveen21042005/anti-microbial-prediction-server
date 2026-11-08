"""
amp_rules.py - rule-based AMP-likeness verdict (no machine learning).

Evidence used
-------------
1. Property rules, tuned on real data (see tune_rules.py). A peptide meets the
   property rule if it has EITHER
       - net charge at pH 7.4 of +2 or more, OR
       - a hydrophobic moment of 0.4 or more.
   Length and hydrophobic-residue fraction are shown for information but are NOT
   scored: in the tuning data they did not help separate DRAMP peptides from
   UniProt non-AMPs.
2. Similarity to known AMPs (optional): a close, high-coverage hit in the
   reference database counts as strong evidence.

Verdict
-------
  "Likely AMP"    - a close known homolog was found
  "Possible AMP"  - no close homolog, but a property rule is met
  "Unlikely AMP"  - otherwise ("does not look like a typical AMP"; this is NOT
                    proof that the peptide is inactive)

How well do the property rules work?
------------------------------------
On a length-matched test half (DRAMP peptides vs presumed-non-AMP UniProt
Swiss-Prot peptides, 10-100 aa), the tuned rule reached about 0.66 balanced
accuracy against all negatives and about 0.71 when ribosomal, histone and
membrane-protein families were excluded. It is a weak indicator: highly
cationic non-AMP proteins will be called "Possible AMP". Re-run
tune_rules.py if you change a threshold or the data.

How well does the similarity evidence work?
-------------------------------------------
See eval_similarity.py. With the 70% identity / 70% coverage cutoff, about 5%
of non-AMPs get a close-homolog hit (false positives). Sensitivity is 75% for
all AMPs but only 43% for AMPs whose nearest other known AMP is below 90%
identical, because DRAMP contains many near-identical variants. A missing
homolog therefore says little about a new peptide, while a close homolog is
fairly strong evidence.
"""
from features import extract_features

# --- Tuned thresholds (edit here, then re-run tune_rules.py) -------------
CHARGE_MIN = 2.0                   # net charge at pH 7.4
MOMENT_MIN = 0.4                   # hydrophobic moment (Eisenberg scale, 100 deg)
HOMOLOG_MIN_IDENTITY = 70.0        # % identity over the aligned region
HOMOLOG_MIN_COVERAGE = 70.0        # % of the query covered
# Measured with eval_similarity.py (300 AMPs + 300 non-AMPs, leave-self-out):
# at identity >= 70 and coverage >= 70, about 5% of non-AMPs fire; 75% of AMPs
# fire overall, but only 43% of AMPs with no near-identical (>=90%) relative.
# Shown for information only (not scored):
TYPICAL_LENGTH = (10, 50)
TYPICAL_HYDROPHOBIC_FRACTION = (0.30, 0.60)
# -------------------------------------------------------------------------


def assess(seq, hits=None):
    """
    seq  : a validated, uppercase sequence
    hits : optional list from similarity.find_similar (best hit first)

    Returns a dict: verdict, property_points, property_max, criteria (list of
    dicts with name / value / rule / points / max_points / met / scored),
    homolog_found, best_hit, hydrophobic_moment, features.
    """
    f = extract_features(seq)
    criteria = []

    charge_met = f["NetCharge"] >= CHARGE_MIN
    criteria.append(
        {"name": "Net charge (pH 7.4)", "value": f"{f['NetCharge']:+.2f}",
         "rule": f">= +{CHARGE_MIN:g}", "points": int(charge_met),
         "max_points": 1, "met": charge_met, "scored": True}
    )

    moment_met = f["HydrophobicMoment"] >= MOMENT_MIN
    criteria.append(
        {"name": "Hydrophobic moment", "value": f"{f['HydrophobicMoment']:.3f}",
         "rule": f">= {MOMENT_MIN:g}", "points": int(moment_met),
         "max_points": 1, "met": moment_met, "scored": True}
    )

    lo, hi = TYPICAL_LENGTH
    criteria.append(
        {"name": "Length", "value": f"{f['Length']} aa",
         "rule": f"typically {lo}-{hi} aa (not scored)", "points": 0,
         "max_points": 0, "met": lo <= f["Length"] <= hi, "scored": False}
    )

    hl, hh = TYPICAL_HYDROPHOBIC_FRACTION
    frac = f["HydrophobicFraction"]
    criteria.append(
        {"name": "Hydrophobic residues", "value": f"{frac:.0%}",
         "rule": f"typically {hl:.0%}-{hh:.0%} (not scored)", "points": 0,
         "max_points": 0, "met": hl <= frac <= hh, "scored": False}
    )

    property_points = sum(c["points"] for c in criteria)
    property_max = sum(c["max_points"] for c in criteria)

    best = hits[0] if hits else None
    homolog = bool(
        best
        and best["identity_pct"] >= HOMOLOG_MIN_IDENTITY
        and best["coverage_pct"] >= HOMOLOG_MIN_COVERAGE
    )

    if homolog:
        verdict = "Likely AMP"
    elif property_points >= 1:
        verdict = "Possible AMP"
    else:
        verdict = "Unlikely AMP"

    return {
        "verdict": verdict,
        "property_points": property_points,
        "property_max": property_max,
        "criteria": criteria,
        "homolog_found": homolog,
        "best_hit": best,
        "hydrophobic_moment": f["HydrophobicMoment"],
        "features": f,
    }


def print_report(result):
    print(f"Verdict: {result['verdict']}")
    print(f"Property rules met: {result['property_points']}/{result['property_max']}")
    for c in result["criteria"]:
        tag = "scored" if c["scored"] else "info  "
        tick = "yes" if c["met"] else "no "
        print(f"  [{tick}] ({tag}) {c['name']}: {c['value']}  (rule: {c['rule']})")
    b = result["best_hit"]
    if b:
        print(f"Closest known peptide: {b['peptide_id']} "
              f"identity {b['identity_pct']}% coverage {b['coverage_pct']}%")
    else:
        print("Similarity search: not run")


if __name__ == "__main__":
    import argparse
    from features import validate_sequence

    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("--db", default=None,
                    help="path to amp.db to include similarity evidence")
    args = ap.parse_args()

    ok, seq = validate_sequence(args.sequence)
    if not ok:
        raise SystemExit(seq)

    hits = None
    if args.db:
        from similarity import load_reference, find_similar
        hits = find_similar(seq, load_reference(args.db), top_n=1)
    print_report(assess(seq, hits))