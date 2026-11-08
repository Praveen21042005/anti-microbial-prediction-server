"""
AMP-Predict web server (Streamlit).

Run with:   streamlit run app.py

Needs in the same folder: amp.db, features.py, similarity.py, amp_rules.py,
app_helpers.py.  No machine-learning model is used: the verdict combines
(1) similarity to known AMPs and (2) rules tuned on real data.
"""
import math
import os

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_helpers as H
from features import validate_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "amp.db")

# Evaluation numbers shown on the About page. Update them if you re-run the
# tuning scripts (tune_rules.py, eval_similarity.py).
EVAL = {
    "rule_bal_acc_all_neg": "about 0.66",
    "rule_bal_acc_filtered_neg": "about 0.71",
    "sim_fpr_70": "about 5%",
    "sim_sens_all_70": "about 75%",
    "sim_sens_novel_70": "about 43%",
    "near_duplicate_share": "about 59%",
}

EXAMPLES = {
    "Custom": None,
    "Melittin (known AMP)": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "LL-37 (human AMP)": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Designed amphipathic peptide": "KWFKAMKSIAKFIAKE",
    "Insulin A chain (not an AMP)": "GIVEQCCTSICSLYQLENYCN",
    "Acidic control (not an AMP)": "EEEEDDDDEEEEDDDD",
}

VERDICT_COLOR = {"Likely AMP": "#10B981", "Possible AMP": "#F59E0B", "Unlikely AMP": "#6B7280"}
VERDICT_TEXT = {
    "Likely AMP": "A close relative of a known antimicrobial peptide was found in the reference database.",
    "Possible AMP": "No close known relative was found, but the sequence has properties often seen in AMPs "
                    "(positive net charge and/or an amphipathic helix).",
    "Unlikely AMP": "No close known relative was found and the sequence does not show the properties tested. "
                    "This does not prove that the peptide is inactive.",
}

st.set_page_config(
    page_title="AMP-Predict - Antimicrobial Peptide Analysis Server",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .verdict-card { background-color: #F3F4F6; padding: 1rem 1.2rem; border-radius: 8px;
                    border-left: 6px solid #6B7280; margin-bottom: 1rem; }
    .verdict-badge { color: white; padding: 4px 14px; border-radius: 12px; font-weight: 700;
                     font-size: 1.1rem; }
    .stButton>button { background-color: #2563EB; color: white; font-weight: 600;
                       border-radius: 6px; border: none; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #1D4ED8; color: white; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------- resources
@st.cache_resource
def get_reference():
    from similarity import load_reference
    return load_reference(DB_PATH)


@st.cache_data
def get_info():
    return H.db_info(DB_PATH)


@st.cache_data
def get_property_sample():
    return H.db_property_sample(DB_PATH)


# --------------------------------------------------------------------- helpers
def load_example():
    seq = EXAMPLES.get(st.session_state.get("example_choice"))
    if seq:
        st.session_state["seq_input"] = seq


def helical_wheel(seq):
    colors, xs, ys, labels = [], [], [], []
    for i, aa in enumerate(seq):
        angle = math.radians(i * 100 % 360)
        r = 1.0 + i * 0.03
        xs.append(r * math.cos(angle))
        ys.append(r * math.sin(angle))
        labels.append(f"{aa}{i + 1}")
        if aa in "KRH":
            colors.append("#2563EB")
        elif aa in "DE":
            colors.append("#EF4444")
        elif aa in "AVILMFWPG":
            colors.append("#10B981")
        else:
            colors.append("#9CA3AF")
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="markers+text", text=labels, textposition="middle center",
            marker=dict(size=30, color=colors, line=dict(width=1, color="black")),
            showlegend=False,
        )
    )
    span = max(2.0, 1.2 + len(seq) * 0.03)
    fig.update_layout(
        xaxis=dict(visible=False, range=[-span, span]),
        yaxis=dict(visible=False, range=[-span, span], scaleanchor="x"),
        width=480, height=480, margin=dict(l=20, r=20, t=20, b=40),
        annotations=[dict(x=0, y=-span, showarrow=False, font=dict(size=11),
                          text="Blue: basic, Red: acidic, Green: hydrophobic, Grey: polar")],
    )
    return fig


# ----------------------------------------------------------------- page: single
def page_single(reference, info):
    st.subheader("🔍 Analyse a single peptide")
    st.write(
        "Enter an amino acid sequence (5-100 residues, standard one-letter codes, or one FASTA record). "
        "The server searches a library of known antimicrobial peptides and checks properties "
        "typical of AMPs."
    )

    if "seq_input" not in st.session_state:
        st.session_state["seq_input"] = EXAMPLES["Melittin (known AMP)"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.text_area("Peptide sequence:", key="seq_input", height=120)
    with col2:
        st.selectbox("Load a sample:", list(EXAMPLES), key="example_choice", on_change=load_example)
        st.caption("Your sequence is not stored on the server.")

    if st.button("Run analysis"):
        raw = H.sequence_from_input(st.session_state["seq_input"])
        ok, seq_or_err = validate_sequence(raw, H.MIN_LEN, H.MAX_LEN)
        if not ok:
            st.session_state.pop("single", None)
            st.error(f"Input error: {seq_or_err}")
        else:
            with st.spinner(f"Comparing with {len(reference):,} known peptides..."):
                result, hits = H.analyse(seq_or_err, reference, top_n=5)
            st.session_state["single"] = {
                "seq": seq_or_err, "result": result, "hits": hits,
                "report": H.build_report(seq_or_err, result, hits, info),
            }

    data = st.session_state.get("single")
    if data:
        show_single_result(data)


def show_single_result(data):
    seq, result, hits = data["seq"], data["result"], data["hits"]
    feats, verdict = result["features"], result["verdict"]

    st.markdown("---")
    color = VERDICT_COLOR[verdict]
    st.markdown(
        f'<div class="verdict-card" style="border-left-color:{color}">'
        f'<span class="verdict-badge" style="background-color:{color}">{verdict}</span>'
        f"<p style='margin:0.6rem 0 0 0'>{VERDICT_TEXT[verdict]}</p></div>",
        unsafe_allow_html=True,
    )
    if feats["Length"] < 10:
        st.warning("The rules were tuned on peptides of 10-100 residues. Treat results for "
                   "shorter sequences with extra caution.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Length", f"{feats['Length']} aa")
    c2.metric("Net charge (pH 7.4)", f"{feats['NetCharge']:+.2f}")
    c3.metric("Hydrophobic moment", f"{feats['HydrophobicMoment']:.3f}")
    best = result["best_hit"]
    c4.metric("Closest known AMP", f"{best['identity_pct']}% identity" if best else "-")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Why this verdict", "Closest known AMPs", "Properties", "Amino acid composition", "Helical wheel"]
    )
    with t1:
        st.dataframe(H.criteria_table(result), hide_index=True)
        st.caption(
            "The verdict is 'Likely AMP' if a close known relative exists; otherwise 'Possible AMP' if net "
            "charge or hydrophobic moment meets its cutoff; otherwise 'Unlikely AMP'."
        )
    with t2:
        st.dataframe(H.hits_table(hits), hide_index=True)
        st.caption("Local alignment (BLOSUM62). Identity is measured over the aligned region; "
                   "coverage is the share of your sequence that aligns.")
    with t3:
        st.dataframe(H.properties_table(feats), hide_index=True)
    with t4:
        fig = px.bar(
            H.composition_table(feats), x="Amino acid", y="Frequency (%)",
            title="Amino acid composition", color="Frequency (%)", color_continuous_scale="Blues",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig)
    with t5:
        st.write("Idealised alpha-helix viewed from above (100 degrees per residue).")
        st.plotly_chart(helical_wheel(seq))

    st.download_button("📥 Download report (TXT)", data=data["report"],
                       file_name="amp_predict_report.txt", mime="text/plain")

    with st.expander("How to interpret this result"):
        st.markdown(
            "- This is a **rule-based indicator, not a probability** and not proof of antimicrobial activity.\n"
            "- A close known relative is fairly strong evidence. Not having one says little about a new peptide.\n"
            "- 'Unlikely AMP' means the sequence does not look like a typical AMP, not that it is inactive.\n"
            "- Cationic non-AMP proteins can be called 'Possible AMP'. Confirm interesting candidates in the lab."
        )


# ------------------------------------------------------------------ page: batch
def page_batch(reference):
    st.subheader("📦 Batch analysis")
    st.write(f"Paste several sequences in FASTA format or upload a file (up to {H.MAX_BATCH} sequences).")

    text = st.text_area(
        "Multi-FASTA sequences:",
        value=">Melittin\nGIGAVLKVLTTGLPALISWIKRKRQQ\n>Acidic_control\nEEEEDDDDEEEEDDDD\n"
              ">Magainin2\nGIGKFLHSAKKFGKAFVGEIMNS",
        height=200,
    )
    uploaded = st.file_uploader("Or upload a FASTA file (.fasta, .fa, .txt)", type=["fasta", "fa", "txt"])
    if uploaded is not None:
        try:
            text = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("The file could not be read as text. Please upload a plain-text FASTA file.")
            return

    if st.button("Run batch analysis"):
        entries = H.parse_fasta(text)
        if not entries:
            st.error("No sequences found.")
        else:
            if len(entries) > H.MAX_BATCH:
                st.warning(f"Only the first {H.MAX_BATCH} of {len(entries)} sequences will be analysed.")
                entries = entries[: H.MAX_BATCH]
            bar = st.progress(0.0)
            df = H.batch_assess(entries, reference, progress=lambda i, n: bar.progress(i / n))
            bar.empty()
            st.session_state["batch"] = df

    df = st.session_state.get("batch")
    if df is not None:
        st.success(f"Analysed {len(df)} sequence(s).")
        st.dataframe(df, hide_index=True)
        st.download_button("📥 Download results (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="amp_predict_batch.csv", mime="text/csv")


# --------------------------------------------------------------- page: database
def page_database(info):
    st.subheader("📚 Reference database of known AMPs")
    st.write(f"{info['count']:,} peptides ({info.get('dataset', 'DRAMP')}). "
             "Search by ID, name, sequence fragment or activity.")

    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input("Search:", "")
    with c2:
        lo, hi = st.slider("Length (aa)", int(info["min_len"]), int(info["max_len"]),
                           (int(info["min_len"]), int(info["max_len"])))

    df, total = H.search_database(DB_PATH, query, (lo, hi), limit=500)
    st.write(f"{total:,} match(es)" + (" - showing the first 500." if total > 500 else "."))
    st.dataframe(df, hide_index=True)

    st.markdown("### Property space of the reference peptides")
    sample = get_property_sample()
    fig = px.scatter(sample, x="HydrophobicMoment", y="NetCharge", color="length",
                     hover_name="peptide_id", opacity=0.6,
                     title=f"Net charge vs hydrophobic moment ({len(sample):,} random peptides)")
    st.plotly_chart(fig)


# ------------------------------------------------------------------- page: about
def page_about(info):
    st.subheader("📖 About and methods")
    st.markdown(
        f"""
### What this server does
AMP-Predict analyses a peptide sequence in two ways and combines them into a verdict:
**Likely AMP**, **Possible AMP** or **Unlikely AMP**. It does not use machine learning.

1. **Similarity to known AMPs.** The sequence is aligned (local alignment, BLOSUM62, gap open -10,
   gap extend -0.5) against {info['count']:,} known AMPs. A hit with at least 70% identity and 70% coverage
   counts as a close known relative.
2. **Property rules.** Net charge at pH 7.4 of at least +2, or a hydrophobic moment of at least 0.4
   (Eisenberg scale, 100 degrees per residue). Length and hydrophobic fraction are shown for
   information but are not used.

### Data
- **Known AMPs:** DRAMP general AMP set, cleaned (standard residues only, 5-100 aa, exact duplicates
  removed). Database built {info.get('built_on', '')}.
- **Non-AMPs (for tuning and testing only):** reviewed UniProtKB/Swiss-Prot entries of 10-100 aa without
  antimicrobial-related keywords (Antimicrobial, Antibiotic, Fungicide, Antiviral protein, Amphibian
  defense peptide, Defensin, Bacteriocin, Toxin), exact duplicates and sequences identical to a known AMP
  removed, then length-matched to the AMPs. These are *presumed* non-AMPs.

### Features
Length, molecular weight, net charge (Henderson-Hasselbalch with side-chain and terminal pKa values),
isoelectric point (bisection), mean hydrophobicity, hydrophobic moment, residue-class fractions and
amino acid composition.

### How well does it work?
- **Property rules** were tuned on one half of a length-matched AMP / non-AMP set and tested on the other
  half. Balanced accuracy was {EVAL['rule_bal_acc_all_neg']} against all non-AMPs and
  {EVAL['rule_bal_acc_filtered_neg']} when ribosomal, histone and membrane-protein families were excluded.
  This is a weak indicator.
- **Similarity evidence** at the 70% / 70% cutoff: {EVAL['sim_fpr_70']} of non-AMPs matched a known AMP.
  Sensitivity was {EVAL['sim_sens_all_70']} for all AMPs but only {EVAL['sim_sens_novel_70']} for AMPs with
  no near-identical (90% or more) relative, because {EVAL['near_duplicate_share']} of the AMPs have one.

### Limitations
- The verdict is a heuristic indicator, not a calibrated probability.
- Presumed non-AMPs may include peptides with undiscovered activity.
- Highly cationic non-AMP proteins can be called "Possible AMP".
- A predicted AMP is not necessarily active against any particular microbe, and activity is not tested here.
- Results are for peptides of 10-100 residues. Shorter peptides were not evaluated.

### Privacy
Submitted sequences are processed in memory and are not saved by the server.

### References
Add formal citations for: DRAMP, UniProt, the BLOSUM62 matrix, and the Eisenberg hydrophobicity scale.
"""
    )


# -------------------------------------------------------------------------- main
def main():
    st.markdown('<div class="main-header">AMP-Predict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Antimicrobial peptide similarity search and AMP-likeness analysis</div>',
        unsafe_allow_html=True,
    )

    if not os.path.exists(DB_PATH):
        st.error("The reference database amp.db was not found next to app.py. "
                 "Build it with: python build_db.py amp_clean.csv --db amp.db")
        st.stop()
    try:
        reference = get_reference()
    except ImportError:
        st.error("Biopython is required for the similarity search. Install it with: pip install biopython")
        st.stop()
    info = get_info()

    st.sidebar.title("🧬 Navigation")
    page = st.sidebar.radio(
        "Select a function:",
        ["Single sequence analysis", "Batch analysis", "Reference database", "About and methods"],
    )
    st.sidebar.caption(f"Reference library: {info['count']:,} known AMPs")

    if page == "Single sequence analysis":
        page_single(reference, info)
    elif page == "Batch analysis":
        page_batch(reference)
    elif page == "Reference database":
        page_database(info)
    else:
        page_about(info)


main()