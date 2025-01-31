import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from fpdf import FPDF
import tempfile
from datetime import datetime
import seaborn as sns

# -------------------------
# Constants and Unit Configuration
# -------------------------
UNIT_CATEGORIES = {
    'mass': ['mg/mL', 'µg/mL', 'ng/mL', 'g/L', 'kg/m³', 'g/dL', '% (w/v)', 'mg/dL', 'µg/µL'],
    'molar': ['M', 'mM', 'µM', 'nM'],
    'activity': ['U/mL', 'mU/mL', 'kU/L'],
    'microbiology': ['CFU/mL', '×10⁶ cells/mL']
}

CONVERSION_FACTORS = {
    # Mass
    'mg/mL': 1, 'µg/mL': 1000, 'ng/mL': 1e6,
    'g/L': 1, 'kg/m³': 1, 'g/dL': 10,
    '% (w/v)': 10, 'mg/dL': 100, 'µg/µL': 0.001,

    # Molar
    'M': 1, 'mM': 1e3, 'µM': 1e6, 'nM': 1e9,

    # Activity
    'U/mL': 1, 'mU/mL': 1e3, 'kU/L': 0.001,

    # Microbiology
    'CFU/mL': 1, '×10⁶ cells/mL': 1e-6
}

PLATE_ROWS = list('ABCDEFGH')
PLATE_COLS = list(range(1, 13))

# -------------------------
# Helper Functions
# -------------------------
def get_unit_category(unit):
    """Return the category (mass/molar/activity/etc.) of the given unit."""
    for category, units in UNIT_CATEGORIES.items():
        if unit in units:
            return category
    return None

def convert_units(value, from_unit, to_unit, mw=None):
    """
    Convert 'value' from 'from_unit' to 'to_unit'.
    If converting between mass and molar units, 'mw' is required.
    """
    if from_unit == to_unit:
        return value

    from_cat = get_unit_category(from_unit)
    to_cat = get_unit_category(to_unit)

    if not (from_cat and to_cat):
        raise ValueError(f"Invalid units: {from_unit} or {to_unit}")

    # If categories differ, ensure it's mass<->molar with MW provided
    if from_cat != to_cat and not ('mass' in {from_cat, to_cat} and 'molar' in {from_cat, to_cat}):
        raise ValueError(f"Cannot convert {from_unit} ({from_cat}) to {to_unit} ({to_cat})")

    # Handle mass <-> molar
    if 'mass' in {from_cat, to_cat} and 'molar' in {from_cat, to_cat}:
        if not mw:
            raise ValueError("Molecular weight required for mass-molar conversion")

        if from_cat == 'mass':
            # mass -> molar
            return (value / mw) * CONVERSION_FACTORS[to_unit]
        else:
            # molar -> mass
            return (value * mw) / CONVERSION_FACTORS[from_unit]

    # Same category conversion
    return value * (CONVERSION_FACTORS[to_unit] / CONVERSION_FACTORS[from_unit])

# -------------------------
# PDF Generation
# -------------------------
def create_compact_report(
    experiment_name,
    date,
    analyst,
    instrument_id,
    fig_curve,
    fig_heatmap,
    qc_metrics,
    plate_layout_df,
    notes
):
    """
    Generate and return a PDF report (as bytes) with the calibration curve, heatmap,
    QC metrics, final concentrations, etc.
    """
    pdf = FPDF(orientation='P')  # Portrait format
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)

    # --- Header ---
    pdf.cell(0, 10, f"{experiment_name}, colorimetric assay", 0, 1, 'C')
    pdf.set_font("", '', 10)
    pdf.cell(0, 6, f"Date: {date} | Analyst: {analyst} | Instrument ID: {instrument_id}", 0, 1, 'C')

    # --- Main content area ---
    pdf.set_y(25)

    # --- Calibration Curve (Left) ---
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # Use only "w=..." so aspect ratio is preserved
        fig_curve.savefig(tmp.name, bbox_inches='tight', dpi=300)
        pdf.image(tmp.name, x=10, y=40, w=80)

    # --- QC Metrics (Right) ---
    pdf.set_xy(90, 40)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(80, 6, "Quality Control Metrics:", 0, 1)
    pdf.set_font("", '', 8)

    qc_items = [
        ("R² Value", qc_metrics['R² Value']),
        ("Regression Equation", qc_metrics['Regression Equation']),
        ("Linear Range", qc_metrics['Linear Range']),
        ("Dilution Factor", qc_metrics['Dilution Factor']),
    ]

    pdf.set_x(90)
    for metric, value in qc_items:
        # Each line in the same x -> align nicely
        current_x = pdf.get_x()
        current_y = pdf.get_y()
        pdf.multi_cell(80, 6, f"{metric}: {value}")
        pdf.set_xy(current_x, current_y + 6)

    # --- Notes section (below the table) ---
    pdf.set_xy(155, 40)
    pdf.set_font("", 'I', 8)
    pdf.multi_cell(40, 6, f"Notes: {notes}")

    # --- Heatmap (Right Calibration Curve) ---
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        fig_heatmap.savefig(tmp.name, bbox_inches='tight', dpi=300)
        pdf.image(tmp.name, x=10, y=105, h=80)

    # --- Concentration Table (Right of Heatmap) ---
    pdf.set_xy(10, 190)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 6, "Final Concentrations:", 0, 1)

    # Create plate layout table
    pdf.set_y(200)
    col_widths = [10] + [12] * 12
    row_height = 6  # a bit smaller for less overlap

    # Table headers
    pdf.set_font("", 'B', 7)
    pdf.set_x(10)
    pdf.cell(col_widths[0], row_height, "", 1)  # Empty cell for row labels
    for col in range(1, 13):
        pdf.cell(col_widths[col], row_height, str(col), 1, 0, 'C')
    pdf.ln(row_height)

    # Table rows
    pdf.set_font("", '', 7)
    for row_idx, row in enumerate(plate_layout_df.values):
        pdf.set_x(10)
        pdf.cell(col_widths[0], row_height, plate_layout_df.index[row_idx], 1)
        for value in row:
            text_val = f"{value:.2f}" if not np.isnan(value) else ""
            pdf.cell(col_widths[1], row_height, text_val, 1, 0, 'C')
        pdf.ln(row_height)

    # Return the PDF as bytes
    return pdf.output(dest='S').encode('latin1')

# -------------------------
# Main Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="96-Well Plate Analyzer", layout="centered")
    st.title("🧪 96-Well Plate Colorimetric Assay Analyzer")

    # --- Session State Defaults ---
    if 'std_unit' not in st.session_state:
        st.session_state.std_unit = 'mg/mL'
    if 'result_unit' not in st.session_state:
        st.session_state.result_unit = 'mg/mL'

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.std_unit = st.selectbox(
            "Standard Unit",
            CONVERSION_FACTORS.keys(),
            index=list(CONVERSION_FACTORS.keys()).index(st.session_state.std_unit)
        )
        st.session_state.result_unit = st.selectbox(
            "Result Unit",
            CONVERSION_FACTORS.keys(),
            index=list(CONVERSION_FACTORS.keys()).index(st.session_state.result_unit)
        )

        # Molecular Weight (if converting mass <-> molar)
        mw = None
        from_cat = get_unit_category(st.session_state.std_unit)
        to_cat = get_unit_category(st.session_state.result_unit)
        if ('mass' in {from_cat, to_cat}) and ('molar' in {from_cat, to_cat}):
            mw = st.number_input("Molecular Weight (g/mol)", min_value=0.1, value=1.0)

        dilution_factor = st.number_input("Dilution Factor", value=1.0, min_value=0.1, step=0.1)
        experiment_name = st.text_input("Experiment Name", "My Assay")
        analyst = st.text_input("Analyst Name", "John Doe")
        instrument_id = st.text_input("Instrument ID", "SynergyMix_01")
        notes = st.text_area("Additional Notes")

    # --- File Upload & Basic Processing ---
    uploaded_file = st.file_uploader("📤 Upload Plate Data CSV", type=["csv"])
    if not uploaded_file:
        return

    try:
        plate_df = pd.read_csv(uploaded_file, header=None)
        plate_df.columns = PLATE_COLS
        plate_df.index = PLATE_ROWS

        df = plate_df.stack().reset_index()
        df.columns = ['Row', 'Column', 'Absorbance']
        df['Well'] = df['Row'] + df['Column'].astype(str)
    except Exception as e:
        st.error(f"❌ File Error: {str(e)}")
        return

    # Shortcuts
    std_unit = st.session_state.std_unit
    result_unit = st.session_state.result_unit

    # --- Plate Layout Editor (Standards) ---
    st.header("🔬 Plate Layout")
    with st.expander("Edit Standard Concentrations"):
        grid_data = pd.DataFrame(np.nan, index=PLATE_ROWS, columns=PLATE_COLS)
        edited_grid = st.data_editor(grid_data, height=400, use_container_width=True)

    # Merge standard concentrations into df
    melted_std = edited_grid.stack().reset_index()
    melted_std.columns = ['Row', 'Column', 'Concentration']
    melted_std['Column'] = melted_std['Column'].astype(int)
    df = pd.merge(df, melted_std, on=['Row', 'Column'], how='left')

    # --- Process Standards ---
    standards = df.dropna(subset=['Concentration'])
    if len(standards) < 2:
        st.error("❌ Minimum 2 standards required")
        return

    try:
        X = standards[['Concentration']]
        y = standards['Absorbance']
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
    except Exception as e:
        st.error(f"📉 Regression Error: {str(e)}")
        return

    # --- Display Calibration ---
    col1, col2 = st.columns(2)
    with col1:
        st.header("📈 Calibration")
        fig_calib, ax = plt.subplots()
        ax.scatter(X, y, c='#2c7bb6', edgecolor='w')
        x_range = np.linspace(X.min().iloc[0], X.max().iloc[0], 100)
        ax.plot(x_range, model.predict(x_range.reshape(-1, 1)), 'r--',
                label=f"{equation}\nR² = {r2:.4f}")
        ax.set_xlabel(f"Concentration ({std_unit})")
        ax.legend()
        st.pyplot(fig_calib)

    with col2:
        st.header("✅ Quality Control")
        st.metric("R² Value", f"{r2:.4f}")
        st.metric("Linear Range",
                  f"{X.min().iloc[0]:.2f}-{X.max().iloc[0]:.2f} {std_unit}")
        st.metric("Samples Processed", len(df) - len(standards))

    # --- Sample Calculations ---
    samples = df[df['Concentration'].isna()].copy()
    final_concentrations = None
    if not samples.empty:
        try:
            # Convert absorbance -> raw concentration -> target units
            raw_conc = (samples['Absorbance'] - model.intercept_) / model.coef_[0]
            converted = convert_units(raw_conc, std_unit, result_unit, mw)
            samples['Result'] = converted * dilution_factor
        except Exception as e:
            st.error(f"🧪 Calculation Error: {str(e)}")
            return

        # --- Absorbance Heatmap ---
        st.subheader("🌡️ Absorbance Heatmap")
        fig_heatmap, ax = plt.subplots(figsize=(10, 5))
        heatmap_data = df.set_index(['Row', 'Column'])['Absorbance'].unstack()
        cax = ax.matshow(heatmap_data, cmap='viridis', aspect='auto')

        # Add text annotations
        for (i, j), val in np.ndenumerate(heatmap_data):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='w' if val < heatmap_data.values.mean() else 'k',
                    fontsize=6)

        plt.xticks(range(12), PLATE_COLS)
        plt.yticks(range(8), PLATE_ROWS)
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.colorbar(cax, label='Absorbance')
        st.pyplot(fig_heatmap)

        # --- Display Results ---
        st.header("📋 Results")
        st.dataframe(
            samples[['Well', 'Absorbance', 'Result']]
            .rename(columns={'Result': result_unit}),
            height=300,
            use_container_width=True
        )

        # Generate final concentration matrix
        final_concentrations = pd.pivot_table(
            samples,
            values='Result',
            index='Row',
            columns='Column',
            aggfunc='first',
            fill_value=np.nan
        ).reindex(index=PLATE_ROWS, columns=PLATE_COLS)

    # --- Report Generation ---
    if st.button("📄 Generate Report"):
        try:
            # 1) Calibration Curve Figure (smaller for PDF)
            fig_curve_pdf, ax = plt.subplots(figsize=(4, 3))
            x_range = np.linspace(X.min().iloc[0], X.max().iloc[0], 100)
            y_pred = model.predict(x_range.reshape(-1, 1))

            ax.scatter(standards['Concentration'], standards['Absorbance'], color='#2c7bb6')
            ax.plot(x_range, y_pred, color='red')
            ax.set_xlabel('Concentration')
            ax.set_ylabel('Absorbance')
            ax.set_title('Calibration Curve')

            # 2) Heatmap of final concentrations (smaller, with smaller font to avoid overlap)
            fig_heatmap_pdf, ax = plt.subplots(figsize=(6.75, 3))
            if final_concentrations is not None:
                sns.heatmap(final_concentrations,
                            annot=True,
                            fmt=".2f",
                            cmap='viridis',
                            ax=ax,
                            annot_kws={"size": 6})
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
                ax.set_title('Concentration Matrix')
                ax.set_aspect('auto')
            else:
                # If there were no samples, just show an empty placeholder
                ax.text(0.5, 0.5, "No sample results", ha='center', va='center')
                ax.set_axis_off()

            # QC metrics dictionary
            qc_metrics = {
                'R² Value': f"{r2:.4f}",
                'Regression Equation': equation,
                'Linear Range': f"{X.min().iloc[0]:.2f}-{X.max().iloc[0]:.2f} {std_unit}",
                'Dilution Factor': dilution_factor
            }

            # Generate PDF
            pdf_bytes = create_compact_report(
                experiment_name=experiment_name,
                date=datetime.now().strftime("%Y-%m-%d"),
                analyst=analyst,
                instrument_id=instrument_id,
                fig_curve=fig_curve_pdf,
                fig_heatmap=fig_heatmap_pdf,
                qc_metrics=qc_metrics,
                plate_layout_df=(
                    final_concentrations if final_concentrations is not None
                    else pd.DataFrame(np.nan, index=PLATE_ROWS, columns=PLATE_COLS)
                ),
                notes=notes
            )

            # Download button
            st.download_button(
                "⬇️ Download Report",
                pdf_bytes,
                "assay_report.pdf",
                "application/pdf"
            )
        except Exception as e:
            st.error(f"❌ Report Generation Failed: {str(e)}")

if __name__ == "__main__":
    main()