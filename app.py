import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline
import io
import re
import datetime

# --- WEB APP CONFIGURATION ---
st.set_page_config(page_title="Bio-Oil TAN Analyzer", layout="wide")
st.title("🧪 Bio-Oil TAN Titration Analyzer")

# --- THEORY & LOGIC SECTION ---
with st.expander("ℹ️ Theory & Mathematical Logic"):
    st.markdown("""
    **1. CHEMICAL LOGIC & LITERATURE**
    As defined in Skoog's *Fundamentals of Analytical Chemistry*, the equivalence point in potentiometry occurs at the maximum rate of change of potential per unit volume. Because potential (mV) is inversely proportional to pH (Nernst eq: E = E0 - 0.05916 * pH), adding base causes the mV to drop:
    * The 1st derivative produces a negative peak.
    * The 2nd derivative crosses zero from NEGATIVE to POSITIVE.

    **2. MATHEMATICAL LOGIC (Smoothing & Splines)**
    Raw data is noisy. To fix this:
    * **Cubic Spline:** Fits a continuous mathematical equation to the data.
    * **Smoothing (s):** Acts as a filter. Higher 's' ignores micro-noise.
    * **Analytical Derivatives:** We calculate exact, smooth derivatives from the spline equation to pinpoint the stoichiometric endpoints.
    """)

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Analysis Parameters")
sample_id = st.sidebar.text_input("Sample Name (User ID):", "Sample_001")
weight = st.sidebar.number_input("Sample Weight (W) [g]:", value=1.000, format="%.3f")
blank_vol = st.sidebar.number_input("Blank Volume [mL]:", value=0.000, format="%.3f")
molarity = st.sidebar.number_input("Titrant Molarity (M):", value=0.100, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Sensitivity")
s_val = st.sidebar.number_input("Smoothing (s):", value=400)
gate_val = st.sidebar.number_input("Slope Gate (%):", value=5.0)

# --- MAIN AREA: FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload Titration CSV File", type=["csv"])

if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    v_col = [c for c in df.columns if 'Volume' in c][0]
    mv_col = [c for c in df.columns if 'E' in c or 'mV' in c][0]
    v_raw, mv_raw = df[v_col].values, df[mv_col].values
    
    # Sort data
    idx = np.argsort(v_raw); v, mv = v_raw[idx], mv_raw[idx]

    # 2. Spline & Math
    spline = UnivariateSpline(v, mv, k=3, s=s_val)
    v_fine = np.linspace(v.min(), v.max(), 5000)
    e_f, d1, d2 = spline(v_fine), spline.derivative(n=1)(v_fine), spline.derivative(n=2)(v_fine)

    crossings = []
    for i in range(len(d2) - 1):
        if d2[i] < 0 and d2[i+1] > 0:
            x_zero = v_fine[i] - d2[i] * (v_fine[i+1] - v_fine[i]) / (d2[i+1] - d2[i])
            crossings.append(x_zero)

    gate_decimal = gate_val / 100.0
    max_slope = np.max(np.abs(d1[v_fine > 0.1]))
    ep_vols = [x for x in crossings if x > 0.1 and np.abs(spline.derivative(n=1)(x)) > max_slope * gate_decimal]

    # 3. Results Table
    results = []
    for i, ve in enumerate(ep_vols):
        an = (ve - blank_vol) * molarity * 56.1 / weight
        mv_val = float(spline(ve))
        results.append({"ID": f"EP{i+1}", "Vol (mL)": round(ve, 3), "mV": round(mv_val, 1), "AN (mg/g)": round(an, 2)})
    
    results_df = pd.DataFrame(results)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Calculated Endpoints")
        st.dataframe(results_df, use_container_width=True)
    
    # 4. Plotting
    with col2:
        st.subheader("Titration Curve & Derivatives")
        fig, ax_e = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(right=0.8)
        ax1, ax2 = ax_e.twinx(), ax_e.twinx()
        ax2.spines.right.set_position(("axes", 1.15))

        ax_e.plot(v, mv, 'k.', alpha=0.15)
        ax_e.plot(v_fine, e_f, 'k-', lw=1.5)
        ax1.plot(v_fine, d1, 'r-', alpha=0.5)
        ax2.plot(v_fine, d2, 'b-', alpha=0.3)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)

        for i, ve in enumerate(ep_vols):
            ax_e.axvline(ve, color='magenta', linestyle=':')
            ax_e.text(ve, ax_e.get_ylim()[1], f' EP{i+1}', rotation=90, color='magenta', fontweight='bold', va='top')

        ax_e.set_xlabel("Volume (mL)"); ax_e.set_ylabel("mV")
        ax1.set_ylabel("1st Deriv", color='r'); ax2.set_ylabel("2nd Deriv", color='b')
        st.pyplot(fig)

    # 5. PDF Generation (In-Memory)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig_pdf = plt.figure(figsize=(8.27, 11.69))
        ax_rep = fig_pdf.add_axes([0.10, 0.58, 0.62, 0.32])
        
        ax_rep_1, ax_rep_2 = ax_rep.twinx(), ax_rep.twinx()
        ax_rep_2.spines.right.set_position(("axes", 1.15))
        ax_rep.plot(v, mv, 'k.', alpha=0.15); ax_rep.plot(v_fine, e_f, 'k-', lw=1.5)
        ax_rep_1.plot(v_fine, d1, 'r-', alpha=0.5); ax_rep_2.plot(v_fine, d2, 'b-', alpha=0.3)
        ax_rep_2.axhline(0, color='gray', linestyle='--', alpha=0.3)
        for i, ve in enumerate(ep_vols):
            ax_rep.axvline(ve, color='magenta', linestyle=':')
            ax_rep.text(ve, ax_rep.get_ylim()[1], f' EP{i+1}', rotation=90, color='magenta', fontweight='bold', va='top')
        
        ax_rep.set_title(f"TITRATION ANALYSIS: {sample_id}", pad=25, fontweight='bold')
        meta = (f"SOURCE DATA: {uploaded_file.name}\n"
                f"REPORT DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"WEIGHT: {weight} g | BLANK: {blank_vol} mL\n"
                f"SMOOTHING: {s_val} | GATE: {gate_val}%")
        plt.figtext(0.10, 0.52, meta, fontsize=10, family='monospace', va='top')

        ax_tab = fig_pdf.add_axes([0.1, 0.1, 0.8, 0.3]); ax_tab.axis('off')
        if not results_df.empty:
            tab = ax_tab.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
            tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.5)

        pdf.savefig(fig_pdf); plt.close(fig_pdf)
    
    buffer.seek(0)
    
    # 6. Download Button
    st.markdown("---")
    st.download_button(
        label="📄 Download PDF Report",
        data=buffer,
        file_name=f"{sample_id}_Report.pdf",
        mime="application/pdf"
    )