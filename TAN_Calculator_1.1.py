"""
=============================================================================
BIO-OIL TAN TITRATION ANALYZER (Streamlit Version)
=============================================================================
Description:
Processes potentiometric titration data to determine Carboxylic Acid Number 
(CAN) and Total Acid Number (TAN) following NREL/TP-5100-65890 guidelines. 

Assignment Logic:
1. INFLECTION DETECTION: Endpoints are identified via the 2nd derivative 
   zero-crossings of a cubic smoothing spline.
2. NOISE FILTERING: Peaks in the positive mV range (electrode equilibration)
   are discarded.
3. CAN SELECTION: Assigned to the STRONGEST (highest 1st derivative) peak 
   occurring above the user-defined mV Cutoff (e.g., > -400 mV).
4. TAN SELECTION: Assigned to the FINAL valid peak occurring below the 
   user-defined mV Cutoff (e.g., < -400 mV).
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import re
import datetime
import io

st.set_page_config(page_title="TAN Analyzer v2", layout="wide")

st.title("Titration TAN Analyzer")

with st.expander("ℹ️ Theory & Logic"):
    st.markdown("""
    **1. CHEMICAL LOGIC**
    Equivalence points occur at the maximum rate of change of potential. Because potential (mV) drops as pH rises, the 1st derivative is a negative peak and the 2nd derivative crosses zero from NEGATIVE to POSITIVE.

    **2. CLASSIFICATION LOGIC**
    Bio-oils contain a spectrum of acids. Following NREL LAP guidelines:
    * **CARBOXYLIC WINDOW:** Potentials above the specified Cutoff (e.g. > -400 mV).
    * **PHENOLIC WINDOW:** Potentials below the specified Cutoff (e.g. < -400 mV).

    **3. AUTOMATED ASSIGNMENT**
    * **CAN (Carboxylic Acid Number):** Assigned to the STRONGEST (steepest) peak within the Carboxylic window to ensure the most well-defined carboxylic point is used.
    * **TAN (Total Acid Number):** Assigned to the FINAL peak in the Phenolic window, representing the cumulative neutralization of all acidic species.
    """)

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Analysis Parameters")

sample_name = st.sidebar.text_input("Sample Name:", value="Sample_001")
# Weight is initialized as None to prevent calculation without user input
weight = st.sidebar.number_input("Sample Weight (W) [g]:", value=None, format="%.4f")
blank_vol = st.sidebar.number_input("Blank Volume [mL]:", value=0.000, format="%.3f")
molarity = st.sidebar.number_input("Titrant Molarity (M):", value=0.100, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Sensitivity")
s_val = st.sidebar.number_input("Smoothing (s):", value=100)
gate_val = st.sidebar.number_input("Slope Gate (%):", value=10.0)
mv_cutoff = st.sidebar.number_input("mV Cutoff (CAN/TAN):", value=-400)

# Max Volume Toggle
use_max_vol = st.sidebar.checkbox("Enable Max Vol (mL)")
max_vol_limit = 20.0
if use_max_vol:
    max_vol_limit = st.sidebar.number_input("Cutoff Volume:", value=20.0)

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Choose a titration CSV file", type="csv")

if uploaded_file is not None:
    if weight is None or weight <= 0:
        st.error("Please enter a valid Sample Weight to proceed.")
    else:
        try:
            # 1. Load Data
            df = pd.read_csv(uploaded_file)
            v_col = [c for c in df.columns if 'Volume' in c][0]
            mv_col = [c for c in df.columns if 'E' in c or 'mV' in c][0]
            
            v_raw, mv_raw = df[v_col].values, df[mv_col].values
            idx = np.argsort(v_raw)
            v, mv = v_raw[idx], mv_raw[idx]
            
            # Determine effective data range
            max_v = max_vol_limit if use_max_vol else v.max()

            # 2. Mathematical Processing
            spline = UnivariateSpline(v, mv, k=3, s=s_val)
            v_fine = np.linspace(v.min(), v.max(), 5000)
            e_f = spline(v_fine)
            d1 = spline.derivative(n=1)(v_fine)
            d2 = spline.derivative(n=2)(v_fine)
            
            # Calculate Gate relative to valid range
            valid_d1 = d1[(v_fine > 0.1) & (v_fine <= max_v)]
            max_slope = np.max(np.abs(valid_d1)) if len(valid_d1) > 0 else 1.0

            # 3. Endpoint Identification
            valid_eps = []
            ep_counter = 1
            for i in range(len(d2) - 1):
                if d2[i] < 0 and d2[i+1] > 0:
                    x_zero = v_fine[i] - d2[i] * (v_fine[i+1] - v_fine[i]) / (d2[i+1] - d2[i])
                    strength = np.abs(spline.derivative(n=1)(x_zero))
                    
                    if 0.1 < x_zero <= max_v and strength > max_slope * (gate_val/100.0):
                        mv_val = float(spline(x_zero))
                        # Only accept peaks in negative potential region
                        if mv_val < 0:
                            an = (x_zero - blank_vol) * molarity * 56.1 / weight
                            valid_eps.append({
                                "ID": f"EP{ep_counter}",
                                "Vol": x_zero,
                                "mV": mv_val,
                                "Strength": strength,
                                "AN": an
                            })
                            ep_counter += 1

            # 4. Classification Logic
            can_ep, tan_ep, max_can_s = None, None, -1
            for ep in valid_eps:
                if ep["mV"] >= mv_cutoff:
                    if ep["Strength"] > max_can_s:
                        max_can_s, can_ep = ep["Strength"], ep
                else:
                    tan_ep = ep

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax_d1 = ax.twinx()
                ax_d2 = ax.twinx()
                ax_d2.spines.right.set_position(("axes", 1.12))

                ax.plot(v, mv, 'k.', alpha=0.1, label="Raw Data")
                ax.plot(v_fine, e_f, 'k-', lw=1.5, label="Spline")
                ax_d1.plot(v_fine, d1, 'r-', alpha=0.4, label="1st Deriv")
                ax_d2.plot(v_fine, d2, 'b-', alpha=0.2, label="2nd Deriv")
                
                ax.axhline(mv_cutoff, color='orange', ls='--', alpha=0.6)
                if use_max_vol:
                    ax.axvline(max_v, color='red', ls='--', alpha=0.5)

                for ep in valid_eps:
                    ax.axvline(ep["Vol"], color='magenta', ls=':', alpha=0.8)
                    lbl = f' {ep["ID"]}'
                    if ep == can_ep: lbl += " (CAN)"
                    if ep == tan_ep: lbl += " (TAN)"
                    ax.text(ep["Vol"], ax.get_ylim()[1], lbl, rotation=90, weight='bold', color='magenta', va='top')

                ax.set_xlabel("Volume (mL)")
                ax.set_ylabel("mV")
                ax_d1.set_ylabel("1st Deriv (Slope)", color='r')
                ax_d2.set_ylabel("2nd Deriv", color='b')
                st.pyplot(fig)

            with col2:
                st.subheader("Detected Endpoints")
                table_data = []
                for ep in valid_eps:
                    assign = "CAN" if ep == can_ep else ("TAN" if ep == tan_ep else "")
                    table_data.append({
                        "ID": ep["ID"],
                        "Vol": round(ep["Vol"], 3),
                        "mV": round(ep["mV"], 1),
                        "mg/g": round(ep["AN"], 2),
                        "Class": assign
                    })
                st.table(pd.DataFrame(table_data))
                
                if can_ep:
                    st.success(f"**Final CAN:** {can_ep['AN']:.2f} mg KOH/g")
                if tan_ep:
                    st.info(f"**Final TAN:** {tan_ep['AN']:.2f} mg KOH/g")

        except Exception as e:
            st.error(f"Error processing data: {e}")
