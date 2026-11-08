import math
 
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
 
AMINO_ACID_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "E": 147.13, "Q": 146.15, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
 
# Eisenberg consensus hydrophobicity scale
HYDROPHOBICITY = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "E": -0.74, "Q": -0.85, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
 
PKA_VALUES = {
    "Cterm": 3.55, "Nterm": 7.59,
    "C": 9.00, "D": 4.05, "E": 4.45, "H": 5.98, "K": 10.00, "R": 12.00, "Y": 10.00,
}

HYDROPHOBIC_RESIDUES = set("AILMFVM")

def validate_sequence(seq, min_len=5, max_len=100):
    seq = seq.upper().strip().replace(" ", "")
    if not seq:
        return False, "Sequence is empty."
    invalid = set(seq) - set(AMINO_ACIDS)
    if invalid:
        return False, f"Invalid amino acids detected: {', '.join(sorted(invalid))}"
    if len(seq) < min_len or len(seq) > max_len:
        return False, f"Sequence length must be between {min_len} and {max_len} amino acids."
    return True, seq
 
 
def calculate_net_charge(seq, ph=7.4):
    n_term = 1.0 / (1.0 + 10 ** (ph - PKA_VALUES["Nterm"]))
    k = seq.count("K") / (1.0 + 10 ** (ph - PKA_VALUES["K"]))
    r = seq.count("R") / (1.0 + 10 ** (ph - PKA_VALUES["R"]))
    h = seq.count("H") / (1.0 + 10 ** (ph - PKA_VALUES["H"]))
 
    c_term = -1.0 / (1.0 + 10 ** (PKA_VALUES["Cterm"] - ph))
    d = -seq.count("D") / (1.0 + 10 ** (PKA_VALUES["D"] - ph))
    e = -seq.count("E") / (1.0 + 10 ** (PKA_VALUES["E"] - ph))
    c = -seq.count("C") / (1.0 + 10 ** (PKA_VALUES["C"] - ph))
    y = -seq.count("Y") / (1.0 + 10 ** (PKA_VALUES["Y"] - ph))
    return n_term + k + r + h + c_term + d + e + c + y
 
 
def calculate_isoelectric_point(seq):
    lo, hi = 0.0, 14.0
    while (hi - lo) > 0.01:
        mid = (lo + hi) / 2.0
        if calculate_net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 2)
 
 
def calculate_hydrophobic_moment(seq, angle=100):
    rad = math.radians(angle)
    s = sum(HYDROPHOBICITY.get(aa, 0.0) * math.sin(i * rad) for i, aa in enumerate(seq))
    c = sum(HYDROPHOBICITY.get(aa, 0.0) * math.cos(i * rad) for i, aa in enumerate(seq))
    return math.sqrt(s ** 2 + c ** 2) / len(seq)
 
 
def extract_features(seq):
    """Same keys as the original app, plus HydrophobicFraction."""
    n = len(seq)
    mw = sum(AMINO_ACID_MASS.get(aa, 0.0) for aa in seq) - (n - 1) * 18.015
    feats = {
        "Length": n,
        "MW_Da": round(mw, 2),
        "NetCharge": round(calculate_net_charge(seq), 2),
        "pI": calculate_isoelectric_point(seq),
        "AvgHydrophobicity": round(sum(HYDROPHOBICITY[a] for a in seq) / n, 3),
        "HydrophobicMoment": round(calculate_hydrophobic_moment(seq), 3),
        "HydrophobicFraction": round(sum(a in HYDROPHOBIC_RESIDUES for a in seq) / n, 3),
        "CationicRatio": round((seq.count("K") + seq.count("R") + seq.count("H")) / n, 3),
        "AnionicRatio": round((seq.count("D") + seq.count("E")) / n, 3),
        "AliphaticRatio": round(sum(seq.count(a) for a in "AILV") / n, 3),
        "AromaticRatio": round(sum(seq.count(a) for a in "FWY") / n, 3),
    }
    feats.update({f"AAC_{aa}": (seq.count(aa) / n) * 100 for aa in AMINO_ACIDS})
    return feats