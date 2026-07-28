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
2. NOISE FILTERING: Peaks before the user-defined Start Volume are discarded.
3. SHOCK DETECTION: Warns the user if the strongest peak occurs immediately 
   at the start of the titration (signature of electrode shock).
4. CAN SELECTION: Assigned to the STRONGEST (highest 1st derivative) peak 
   occurring within the empirically defined CAN mV Window.
5. TAN SELECTION: Assigned to the FINAL valid peak occurring within the 
   emperically-defined TAN mV Window.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

st.set_page_config(page_title="TAN Analyzer v2", layout="wide")

st.title("Titration TAN Analyzer")

with st.expander("ℹ️ Theory & Logic"):
    st.markdown("""
    **1. CHEMICAL LOGIC**
    Equivalence points occur at the maximum rate of change of potential. Because potential (mV) drops as pH rises, the 1st derivative is a negative peak and the 2nd derivative crosses zero from NEGATIVE to POSITIVE.

    **2. CLASSIFICATION LOGIC**
    Bio-oils contain a spectrum of acids. Following NREL LAP guidelines:
    * **CARBOXYLIC WINDOW:** Captures aliphatic and multifunctional carboxylic acids (+150 to -320 mV).
    * **PHENOLIC WINDOW:** Captures weak and sterically hindered phenolic groups (-350 to -480 mV).

    **3. AUTOMATED ASSIGNMENT**
    * **CAN (Carboxylic Acid Number):** Assigned to the STRONGEST (steepest) peak within the Carboxylic window. Features a Smart Fallback to capture strong early peaks if they exceed the CAN ceiling.
    * **TAN (Total Acid Number):** Assigned to the FINAL peak in the Phenolic window, representing the cumulative neutralization of all acidic species.
    """)

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Analysis Parameters")

with st.sidebar.expander("ℹ️ Theory & Logic"):
    st.markdown("""
    **1. CHEMICAL LOGIC**
    Equivalence points occur at the maximum rate of change of potential. Because potential (mV) drops as pH rises, the 1st derivative is a negative peak and the 2nd derivative crosses zero from NEGATIVE to POSITIVE.

    **2. CLASSIFICATION LOGIC**
    Bio-oils contain a spectrum of acids. Following NREL LAP guidelines:
    * **CARBOXYLIC WINDOW:** Captures aliphatic and multifunctional carboxylic acids (+150 to -320 mV).
    * **PHENOLIC WINDOW:** Captures weak and sterically hindered phenolic groups (-350 to -480 mV).

    **3. AUTOMATED ASSIGNMENT**
    * **CAN (Carboxylic Acid Number):** Assigned to the STRONGEST (steepest) peak within the Carboxylic window to ensure the most well-defined carboxylic point is used. Features a Smart Fallback to capture strong early peaks if they exceed the CAN ceiling.
    * **TAN (Total Acid Number):** Assigned to the FINAL peak in the Phenolic window, representing the cumulative neutralization of all acidic species.
    """)

sample_name = st.sidebar.text_input("Sample Name:", value="Sample_001")
weight = st.sidebar.number_input("Sample Weight (W) [g]:", value=None, format="%.4f")
blank_vol = st.sidebar.number_input("Blank Volume [mL]:", value=0.000, format="%.3f")
molarity = st.sidebar.number_input("Titrant Molarity (M):", value=0.100, format="%.3f")

# Max Volume Toggle
use_max_vol = st.sidebar.checkbox("Enable Max Vol (mL)")
max_vol_limit = 20.0
if use_max_vol:
    max_vol_limit = st.sidebar.number_input("Cutoff Volume:", value=20.0, format="%.1f")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Sensitivity")

sens_mode = st.sidebar.selectbox("Sensitivity Mode:", ["High", "Medium", "Low", "Manual"], index=1)
is_manual = (sens_mode == "Manual")

if sens_mode == "High":
    def_s, def_gate = 20, 5.0
elif sens_mode == "Medium":
    def_s, def_gate = 100, 10.0
elif sens_mode == "Low":
    def_s, def_gate = 200, 15.0
else:
    def_s, def_gate = 100, 10.0

start_vol = st.sidebar.number_input("Start Volume (mL):", value=0.1, format="%.1f")
s_val = st.sidebar.number_input("Smoothing (s):", value=def_s, disabled=not is_manual)
gate_val = st.sidebar.number_input("Slope Gate (%):", value=def_gate, disabled=not is_manual)

st.sidebar.markdown("---")
st.sidebar.subheader("mV Windows")
unlock_win = st.sidebar.checkbox("Unlock Windows")

can_max = st.sidebar.number_input("CAN Max mV:", value=150.0, disabled=not unlock_win)
can_min = st.sidebar.number_input("CAN Min mV:", value=-320.0, disabled=not unlock_win)
tan_max = st.sidebar.number_input("TAN Max mV:", value=-350.0, disabled=not unlock_win)
tan_min = st.sidebar.number_input("TAN Min mV:", value=-480.0, disabled=not unlock_win)


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
            
            # Apply dynamic START_VOL
            valid_d1 = d1[(v_fine > start_vol) & (v_fine <= max_v)]
            max_slope = np.max(np.abs(valid_d1)) if len(valid_d1) > 0 else 1.0

            # 3. Endpoint Identification
            valid_eps = []
            ep_counter = 1
            for i in range(len(d2) - 1):
                if d2[i] < 0 and d2[i+1] > 0:
                    x_zero = v_fine[i] - d2[i] * (v_fine[i+1] - v_fine[i]) / (d2[i+1] - d2[i])
                    strength = np.abs(spline.derivative(n=1)(x_zero))
                    
                    if start_vol < x_zero <= max_v and strength > max_slope * (gate_val/100.0):
                        mv_val = float(spline(x_zero))
                        an = (x_zero - blank_vol) * molarity * 56.1 / weight
                        valid_eps.append({
                            "ID": f"EP{ep_counter}",
                            "Vol": x_zero,
                            "mV": mv_val,
                            "Strength": strength,
                            "AN": an
                        })
                        ep_counter += 1

            # --- Electrode Shock Warning Logic ---
            if len(valid_eps) > 1:
                first_ep = valid_eps[0]
                if first_ep["Vol"] < start_vol + 0.3:
                    is_strongest = all(first_ep["Strength"] >= ep["Strength"] for ep in valid_eps)
                    if is_strongest:
                        st.warning(f"**Electrode Shock Warning:** A massive initial drop was detected at {first_ep['Vol']:.3f} mL. This is an indication of electrode equilibration lag, not a true chemical endpoint. Because it has the highest mathematical strength, it is assigned as the CAN. **Fix: Increase the 'Start Volume (mL)' parameter in the sidebar to bypass this artifact.**", icon="⚠️")

            # 4. Classification Logic (with Smart Fallback)
            can_ep, tan_ep, max_can_s = None, None, -1
            
            # Primary CAN Search
            for ep in valid_eps:
                if can_min <= ep["mV"] <= can_max:
                    if ep["Strength"] > max_can_s:
                        max_can_s, can_ep = ep["Strength"], ep
            
            # Smart CAN Fallback
            if can_ep is None:
                for ep in valid_eps:
                    if ep["mV"] > can_max:
                        if ep["Strength"] > max_can_s:
                            max_can_s, can_ep = ep["Strength"], ep
            
            # TAN Search
            for ep in valid_eps:
                if tan_min <= ep["mV"] <= tan_max:
                    tan_ep = ep
                    
            if tan_ep is None and len(valid_eps) > 0:
                st.warning("**TAN Not Detected:** Manual sensitivity adjustment is necessary. Caution is advised for erroneous EPs due to noise.", icon="⚠️")

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
                ax_d2.axhline(0, color='gray', linestyle='--', alpha=0.3)
                
                # Shade the CAN and TAN windows
                ax.axhspan(can_min, can_max, color='green', alpha=0.1, label='CAN Window')
                ax.axhspan(tan_min, tan_max, color='orange', alpha=0.1, label='TAN Window')
                
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
                        "Vol": f'{ep["Vol"]:.3f}',
                        "mV": f'{ep["mV"]:.1f}',
                        "mg/g": f'{ep["AN"]:.2f}',
                        "Class": assign
                    })
                if table_data:
                    st.table(pd.DataFrame(table_data))
                else:
                    st.write("No endpoints detected with current settings.")
                
                if can_ep:
                    fallback_txt = " *(Smart Fallback)*" if can_ep["mV"] > can_max else ""
                    st.success(f"**Final CAN:** {can_ep['AN']:.2f} mg KOH/g{fallback_txt}")
                if tan_ep:
                    st.info(f"**Final TAN:** {tan_ep['AN']:.2f} mg KOH/g")

        except Exception as e:
            st.error(f"Error processing data: {e}")
