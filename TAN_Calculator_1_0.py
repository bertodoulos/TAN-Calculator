#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""
=============================================================================
BIO-OIL TAN TITRATION ANALYZER (GUI Version)
=============================================================================
Description:
This application processes potentiometric titration data to determine the 
Total Acid Number (TAN) of bio-oil samples (e.g., following ASTM D664). 
It provides a graphical interface to load instrument CSV data, tune 
detection parameters, visualize the mathematical derivatives, and export 
a finalized PDF report.

1. Chemical Logic (Nernst Equation):
As defined in Skoog's 'Fundamentals of Analytical Chemistry', the equivalence
point in potentiometry occurs at the maximum rate of change of potential.
In standard potentiometric acid-base titrations, adding base increases the pH. 
According to the Nernst equation (E = E0 - 0.05916 * pH), the measured 
potential (mV) is inversely proportional to pH. Because the mV curve 
decreases as the titration progresses:
  • The 1st derivative produces a negative peak (a valley).
  • The 2nd derivative crosses zero from NEGATIVE to POSITIVE.

2. Mathematical Logic (Smoothing & Splines):
Raw titration data is inherently noisy, making standard point-to-point 
(finite difference) derivatives erratic. To fix this:
  • Cubic Spline: Fits a continuous polynomial curve to the data.
  • Smoothing Factor (s): Acts as a noise filter, ignoring micro-
    fluctuations while preserving the true inflection of the curve.
  • Analytical Derivatives: Calculates exact, continuous 1st and 2nd 
    derivatives from the spline equation to pinpoint the stoichiometric 
    endpoints without jagged baseline interference.
=============================================================================
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline
import os
import re

class BioOilAnalyzer:
    def __init__(self, root):
        """Sets up the main GUI window and initializes the control/plot panels."""
        self.root = root
        self.root.title("Titration TAN Analyzer")
        self.root.geometry("1200x900")

        self.filepath = None
        self.analysis_data = {}

        # --- UI LAYOUT: CONTROL PANEL ---
        ctrl_frame = tk.LabelFrame(root, text="Analysis Parameters", padx=10, pady=10)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Standard Sample Metadata Inputs
        self.create_input(ctrl_frame, "Sample Name (User ID):", "ent_name", 0)
        self.create_input(ctrl_frame, "Sample Weight (W) [g]:", "ent_weight", 1)
        self.create_input(ctrl_frame, "Blank Volume [mL]:", "ent_blank", 2, "0.0")
        self.create_input(ctrl_frame, "Titrant Molarity (M):", "ent_molarity", 3, "0.1")

        # Mathematical Tuning Parameters for Endpoint Detection
        tk.Label(ctrl_frame, text="--- Detection Sensitivity ---", fg="blue").grid(row=4, column=0, columnspan=2, pady=(15,5))
        self.create_input(ctrl_frame, "Smoothing (s):", "ent_s", 5, "400")
        self.create_input(ctrl_frame, "Slope Gate (%):", "ent_gate", 6, "5.0")

        # Application Command Buttons
        self.btn_info = tk.Button(ctrl_frame, text="ℹ️ Theory & Logic", command=self.show_theory_info, bg="#E8E8E8")
        self.btn_info.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(15, 5))

        self.btn_load = tk.Button(ctrl_frame, text="1. Select Instrument CSV", command=self.select_file, bg="#D1E8FF")
        self.btn_load.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(5, 2))

        # New Label for File Name Display
        self.lbl_filename = tk.Label(ctrl_frame, text="No file selected", fg="gray", font=("Arial", 9, "italic"))
        self.lbl_filename.grid(row=9, column=0, columnspan=2, pady=(0, 10))

        self.btn_run = tk.Button(ctrl_frame, text="2. Run Analysis", command=self.analyze, bg="#D1FFD1", state=tk.DISABLED)
        self.btn_run.grid(row=10, column=0, columnspan=2, sticky="ew", pady=5)

        self.btn_save = tk.Button(ctrl_frame, text="3. Export PDF Report", command=self.save_to_pdf, bg="#FFFFD1", state=tk.DISABLED)
        self.btn_save.grid(row=11, column=0, columnspan=2, sticky="ew", pady=5)

        # Results Display Table (Now includes mV)
        self.tree = ttk.Treeview(ctrl_frame, columns=("ID", "Vol", "mV", "AN"), show='headings', height=10)
        self.tree.heading("ID", text="ID")
        self.tree.heading("Vol", text="Vol (mL)")
        self.tree.heading("mV", text="mV")
        self.tree.heading("AN", text="mg KOH/g")
        self.tree.column("ID", width=40)
        self.tree.column("Vol", width=80)
        self.tree.column("mV", width=70)
        self.tree.column("AN", width=80)
        self.tree.grid(row=12, column=0, columnspan=2, pady=10)

        # --- UI LAYOUT: PLOTTING AREA ---
        self.plot_frame = tk.Frame(root, bg="white")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_input(self, parent, label, var_name, row, default=""):
        """Utility function to create a label and entry pair in the grid."""
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        entry = tk.Entry(parent)
        entry.insert(0, default)
        entry.grid(row=row, column=1, pady=3)
        setattr(self, var_name, entry)

    def show_theory_info(self):
        """Displays a message box explaining the program description, chemical, and mathematical logic."""
        info_text = (
            "PROGRAM DESCRIPTION\n"
            "This application processes potentiometric titration data to determine "
            "the Total Acid Number (TAN) of bio-oil samples (e.g., ASTM D664). "
            "It loads instrument data, identifies exact equivalence points, and "
            "calculates Acid Numbers (mg KOH/g) using advanced curve smoothing.\n\n"

            "1. CHEMICAL LOGIC & LITERATURE\n"
            "As defined in Skoog's 'Fundamentals of Analytical Chemistry', the equivalence "
            "point in potentiometry occurs at the maximum rate of change of potential "
            "per unit volume of titrant. Because potential (mV) is inversely proportional "
            "to pH (Nernst eq: E = E0 - 0.05916 * pH), adding base causes the mV to drop:\n"
            "• The 1st derivative produces a negative peak (maximum negative slope).\n"
            "• The 2nd derivative crosses zero from NEGATIVE to POSITIVE.\n\n"

            "2. MATHEMATICAL LOGIC (Smoothing & Splines)\n"
            "Raw data is noisy, making standard point-to-point math erratic.\n"
            "• Cubic Spline: Fits a continuous mathematical equation to your data.\n"
            "• Smoothing (s): Acts as a filter. Higher 's' ignores micro-noise.\n"
            "• Analytical Derivatives: We calculate exact, smooth derivatives from "
            "the spline equation, ensuring highly accurate endpoint detection."
        )
        messagebox.showinfo("Theory & Logic", info_text)

    def select_file(self):
        """Opens a file dialog to select the titration CSV data."""
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            # Update the UI with the selected filename
            fname = os.path.basename(self.filepath)
            self.lbl_filename.config(text=f"Selected: {fname}", fg="green")
            self.btn_run.config(state=tk.NORMAL)

    def analyze(self):
        """Processes titration data using cubic splines and analytical derivatives."""
        try:
            # Retrieve numerical inputs
            W = float(self.ent_weight.get())
            V_blk = float(self.ent_blank.get())
            M = float(self.ent_molarity.get())
            S_VAL = float(self.ent_s.get())
            GATE = float(self.ent_gate.get()) / 100.0

            # Load CSV and identify Volume and Potential columns
            df = pd.read_csv(self.filepath)
            v_col = [c for c in df.columns if 'Volume' in c][0]
            mv_col = [c for c in df.columns if 'E' in c or 'mV' in c][0]
            v, mv = df[v_col].values, df[mv_col].values

            # Ensure data is sorted by volume for spline calculation
            idx = np.argsort(v); v, mv = v[idx], mv[idx]

            # Generate cubic spline and calculate 1st and 2nd derivatives
            spline = UnivariateSpline(v, mv, k=3, s=S_VAL)
            v_fine = np.linspace(v.min(), v.max(), 5000)
            e_f, d1, d2 = spline(v_fine), spline.derivative(n=1)(v_fine), spline.derivative(n=2)(v_fine)

            # Locate inflection points via negative-to-positive zero crossings of 2nd derivative
            crossings = []
            for i in range(len(d2) - 1):
                if d2[i] < 0 and d2[i+1] > 0:
                    # Precise root finding via linear interpolation
                    x_zero = v_fine[i] - d2[i] * (v_fine[i+1] - v_fine[i]) / (d2[i+1] - d2[i])
                    crossings.append(x_zero)

            # Filter endpoints based on the slope threshold (Slope Gate)
            max_slope = np.max(np.abs(d1[v_fine > 0.1]))
            ep_vols = [x for x in crossings if x > 0.1 and np.abs(spline.derivative(n=1)(x)) > max_slope * GATE]

            self.analysis_data = {'v':v, 'mv':mv, 'v_f':v_fine, 'e_f':e_f, 'd1':d1, 'd2':d2, 'ep_vols':ep_vols}

            # Populate the results table and calculate Acid Number (mg KOH/g)
            for i in self.tree.get_children(): self.tree.delete(i)
            self.table_rows = []
            for i, ve in enumerate(ep_vols):
                an = (ve - V_blk) * M * 56.1 / W

                # Fetch the exact mV value from the spline at the endpoint volume
                mv_val = float(spline(ve))

                # Format row to include mV
                row = [f"EP{i+1}", f"{ve:.3f}", f"{mv_val:.1f}", f"{an:.2f}"]
                self.tree.insert("", tk.END, values=row)
                self.table_rows.append(row)

            self.show_plot(v, mv, v_fine, e_f, d1, d2, ep_vols)
            self.btn_save.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_plot(self, v, mv, v_f, e_f, d1, d2, ep_vols):
        """Clears existing plot and draws new titration visualization in the GUI."""
        for widget in self.plot_frame.winfo_children(): widget.destroy()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(right=0.8) # Adjusted margin for triple-axis visibility

        self.draw_titration(ax, v, mv, v_f, e_f, d1, d2, ep_vols)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, self.plot_frame).update()
        plt.close(fig)

    def draw_titration(self, ax_e, v, mv, v_f, e_f, d1, d2, ep_vols):
        """Plots mV data on primary axis and derivatives on twin secondary axes."""
        ax1, ax2 = ax_e.twinx(), ax_e.twinx()
        ax2.spines.right.set_position(("axes", 1.15)) # Offset the blue axis further right

        ax_e.plot(v, mv, 'k.', alpha=0.1) # Data points
        ax_e.plot(v_f, e_f, 'k-', lw=1.5) # Spline curve
        ax1.plot(v_f, d1, 'r-', alpha=0.5) # 1st derivative (Slope)
        ax2.plot(v_f, d2, 'b-', alpha=0.3) # 2nd derivative (Curvature)

        ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
        for i, ve in enumerate(ep_vols):
            ax_e.axvline(ve, color='magenta', linestyle=':')
            ax_e.text(ve, ax_e.get_ylim()[1], f' EP{i+1}', rotation=90, color='magenta', fontweight='bold', va='top')

        ax_e.set_ylabel("mV")
        ax1.set_ylabel("1st Deriv", color='r')
        ax2.set_ylabel("2nd Deriv", color='b')
        ax_e.set_xlabel("Volume (mL)")

    def save_to_pdf(self):
        """Exports report with a smarter filename and more compact plot, without a success popup."""
        import datetime

        csv_name = os.path.basename(self.filepath)

        # 1. Date Detection Logic
        date_match = re.search(r'\d{6,8}', csv_name)
        if date_match:
            date_id = date_match.group()
        else:
            date_id = "no_date"

        # Clean up sample name
        clean_sample_name = re.sub(r'[^a-zA-Z0-9]', '_', self.ent_name.get())
        if not clean_sample_name: clean_sample_name = "UnnamedSample"

        suggested = f"{date_id}_{clean_sample_name}.pdf"

        path = filedialog.asksaveasfilename(initialfile=suggested, defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path: return

        try:
            with PdfPages(path) as pdf:
                # Standard A4 page
                fig = plt.figure(figsize=(8.27, 11.69)) 

                # Compact Plot Placement (leaves room for right axis)
                ax_rep = fig.add_axes([0.10, 0.58, 0.62, 0.32]) 

                d = self.analysis_data
                self.draw_titration(ax_rep, d['v'], d['mv'], d['v_f'], d['e_f'], d['d1'], d['d2'], d['ep_vols'])
                ax_rep.set_title(f"TITRATION ANALYSIS: {self.ent_name.get()}", pad=25, fontweight='bold')

                # Metadata text
                meta = (f"SOURCE DATA: {csv_name}\n"
                        f"REPORT DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                        f"SAMPLE ID: {self.ent_name.get()}\n"
                        f"WEIGHT: {self.ent_weight.get()} g | BLANK: {self.ent_blank.get()} mL\n"
                        f"SMOOTHING (s): {self.ent_s.get()} | GATE: {self.ent_gate.get()}%")
                plt.figtext(0.10, 0.52, meta, fontsize=10, family='monospace', va='top')

                # Results table - Now with 4 columns
                ax_tab = fig.add_axes([0.1, 0.1, 0.8, 0.3]); ax_tab.axis('off')
                tab = ax_tab.table(cellText=self.table_rows, colLabels=["ID", "Vol (mL)", "mV", "AN (mg/g)"], loc='center', cellLoc='center')
                tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.5)

                pdf.savefig(fig); plt.close(fig)

            # Removed the messagebox.showinfo success popup for cleaner UX

        except Exception as e:
            # We keep the error popup so the user actually knows if something failed
            messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = BioOilAnalyzer(root)
    root.mainloop()

