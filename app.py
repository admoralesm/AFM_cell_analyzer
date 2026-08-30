"""
AFM Cell Compression Analyzer - Batch Processing Version
Analyzes multiple cells with Igor AFM data files
Supports cantilever constant editing, baseline correction, and mechanical property extraction
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime

from lulevich_model import LulevichModel

# Optional imports
try:
    from baseline_correction import BaselineCorrector, calculate_relative_deformation
    baseline_available = True
except ImportError:
    baseline_available = False

try:
    from igor_parser import IgorParser, load_igor_pair
    igor_available = True
except ImportError:
    igor_available = False

try:
    from google_drive import GoogleDriveManager
    GoogleDriveManager = GoogleDriveManager
except ImportError:
    GoogleDriveManager = None

# Page config
st.set_page_config(
    page_title="AFM Cell Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .section-header {
        color: #2ca02c;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .cell-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'cantilever_constants' not in st.session_state:
    st.session_state.cantilever_constants = {}

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔬 AFM Cell Analyzer</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**v2.0** - Batch Processing")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")

    st.markdown("### Cell Parameters")

    cell_height_um = st.number_input(
        "Cell Height (μm)",
        min_value=1.0,
        max_value=50.0,
        value=8.09,
        step=0.01
    )
    cell_height_m = cell_height_um * 1e-6

    st.markdown("### Cantilever Settings")

    default_cantilever = st.number_input(
        "Default Cantilever Constant (pN/nm)",
        min_value=0.01,
        max_value=10.0,
        value=0.1,
        step=0.01,
        help="Spring constant of cantilever (pN/nm). Will apply to all cells unless overridden."
    )

    st.markdown("### Baseline Correction")

    baseline_method = st.radio(
        "Baseline Method",
        ["Flat", "Linear"],
        help="Flat: constant offset | Linear: fitted slope for slanted baseline"
    )

    manual_baseline = st.checkbox("Manual Baseline Adjustment", value=False)

    st.markdown("### Fitting Parameters")

    fitting_mode = st.radio(
        "Fitting Method",
        ["Auto Detect", "Manual Range"],
    )

    if fitting_mode == "Manual Range":
        col1, col2 = st.columns(2)
        with col1:
            eps_min = st.number_input("ε min", min_value=0.0, max_value=0.5, value=0.02, step=0.01)
        with col2:
            eps_max = st.number_input("ε max", min_value=0.05, max_value=0.5, value=0.3, step=0.01)
    else:
        eps_min, eps_max = None, None

    st.markdown("### Google Drive")

    if GoogleDriveManager is not None:
        drive_enabled = st.checkbox("Enable Google Drive Storage", value=False)
    else:
        drive_enabled = False
        st.info("ℹ️ Google Drive integration not available in cloud")

# Main content
tabs = st.tabs(["📊 Batch Upload", "📈 Results", "💾 Export", "ℹ️ About"])

with tabs[0]:  # Batch Upload
    st.markdown("### Upload Igor Files (Batch Processing)")

    st.info("""
    📌 **How to use:**
    1. For each cell, upload TWO Igor files (approach and retract curves)
    2. Set cell ID and height
    3. Optionally override cantilever constant per cell
    4. Click "Analyze All Cells"
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Upload Multiple Cells")

        num_cells = st.number_input(
            "Number of cells to analyze",
            min_value=1,
            max_value=20,
            value=2
        )

    cell_configs = []

    for i in range(num_cells):
        with st.expander(f"Cell {i+1}", expanded=(i==0)):
            col1, col2, col3 = st.columns(3)

            with col1:
                cell_id = st.text_input(
                    f"Cell {i+1} ID",
                    value=f"Cell_{i+1:02d}",
                    key=f"cell_id_{i}"
                )

            with col2:
                cell_height = st.number_input(
                    f"Cell {i+1} Height (μm)",
                    min_value=1.0,
                    max_value=50.0,
                    value=cell_height_um,
                    step=0.01,
                    key=f"cell_height_{i}"
                )

            with col3:
                cantilever = st.number_input(
                    f"Cantilever Constant (pN/nm)",
                    min_value=0.01,
                    max_value=10.0,
                    value=default_cantilever,
                    step=0.01,
                    key=f"cantilever_{i}"
                )

            col1, col2 = st.columns(2)

            with col1:
                file1 = st.file_uploader(
                    f"Cell {i+1} - File 1 (Igor .ibw)",
                    type=['ibw'],
                    key=f"file1_{i}",
                    accept_multiple_files=False
                )

            with col2:
                file2 = st.file_uploader(
                    f"Cell {i+1} - File 2 (Igor .ibw)",
                    type=['ibw'],
                    key=f"file2_{i}",
                    accept_multiple_files=False
                )

            if file1 and file2:
                st.success(f"✅ {cell_id}: 2 files ready")

                cell_configs.append({
                    'id': cell_id,
                    'height_um': cell_height,
                    'height_m': cell_height * 1e-6,
                    'cantilever': cantilever,
                    'file1': file1,
                    'file2': file2,
                    'index': i
                })

    st.markdown("---")

    if st.button("🚀 Analyze All Cells", use_container_width=True, type="primary"):
        if len(cell_configs) == 0:
            st.error("❌ No cells configured. Please upload files.")
        else:
            st.session_state.results = {}
            progress_bar = st.progress(0)

            for idx, cell in enumerate(cell_configs):
                progress_bar.progress((idx + 1) / len(cell_configs))

                try:
                    with st.spinner(f"Analyzing {cell['id']}..."):
                        result = analyze_cell(
                            cell,
                            cell_height_m,
                            baseline_method,
                            fitting_mode,
                            eps_min,
                            eps_max
                        )

                        st.session_state.results[cell['id']] = result

                        if result['success']:
                            st.success(f"✅ {cell['id']}: Em = {result['Em_MPa']:.2f} MPa, "
                                     f"Ei = {result['Ei_kPa']:.2f} kPa")
                        else:
                            st.error(f"❌ {cell['id']}: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Error analyzing {cell['id']}: {str(e)}")

            st.success("✅ Batch analysis complete!")


with tabs[1]:  # Results
    if len(st.session_state.results) > 0:
        st.markdown("## Results Summary")

        # Results table
        results_list = []
        for cell_id, result in st.session_state.results.items():
            if result.get('success'):
                results_list.append({
                    'Cell ID': cell_id,
                    'Em (MPa)': f"{result['Em_MPa']:.2f}",
                    'Km (kT)': f"{result['Km_kT']:.1f}",
                    'Ei (kPa)': f"{result['Ei_kPa']:.2f}",
                    'R² (Membrane)': f"{result['r2_membrane']:.4f}",
                    'R² (Cytoskeleton)': f"{result['r2_cytoskeleton']:.4f}"
                })

        if results_list:
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, use_container_width=True)

        # Comparison plots
        st.markdown("### Comparison Plots")

        col1, col2 = st.columns(2)

        with col1:
            if len(results_list) > 0:
                # Young's modulus comparison
                cell_ids = [r['Cell ID'] for r in results_list]
                em_values = [float(r['Em (MPa)']) for r in results_list]
                ei_values = [float(r['Ei (kPa)']) for r in results_list]

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Membrane Young's Modulus", "Cytoskeleton Young's Modulus"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )

                fig.add_trace(
                    go.Bar(x=cell_ids, y=em_values, name="Em (MPa)", marker=dict(color='blue')),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Bar(x=cell_ids, y=ei_values, name="Ei (kPa)", marker=dict(color='orange')),
                    row=1, col=2
                )

                fig.update_yaxes(title_text="Em (MPa)", row=1, col=1)
                fig.update_yaxes(title_text="Ei (kPa)", row=1, col=2)
                fig.update_layout(height=400, showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if len(results_list) > 0:
                # Fit quality comparison
                r2_membrane = [float(r['R² (Membrane)']) for r in results_list]
                r2_cyto = [float(r['R² (Cytoskeleton)']) for r in results_list]

                fig_quality = go.Figure()

                fig_quality.add_trace(go.Scatter(
                    x=cell_ids, y=r2_membrane, name="Membrane (R²)",
                    mode='markers+lines', marker=dict(size=10, color='blue')
                ))

                fig_quality.add_trace(go.Scatter(
                    x=cell_ids, y=r2_cyto, name="Cytoskeleton (R²)",
                    mode='markers+lines', marker=dict(size=10, color='orange')
                ))

                fig_quality.add_hline(y=0.95, line_dash="dash", line_color="red",
                                     annotation_text="R²=0.95 (Good fit)")

                fig_quality.update_layout(
                    title="Fitting Quality Across Cells",
                    xaxis_title="Cell ID",
                    yaxis_title="R² Value",
                    height=400
                )

                st.plotly_chart(fig_quality, use_container_width=True)

    else:
        st.info("👈 Run batch analysis first to view results")


with tabs[2]:  # Export
    if len(st.session_state.results) > 0:
        st.markdown("## Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # JSON export
            json_str = json.dumps(st.session_state.results, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                json_str,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            # CSV export
            results_list = []
            for cell_id, result in st.session_state.results.items():
                if result.get('success'):
                    results_list.append({
                        'Cell ID': cell_id,
                        'Em (MPa)': result['Em_MPa'],
                        'Km (kT)': result['Km_kT'],
                        'Ei (kPa)': result['Ei_kPa'],
                        'R2_Membrane': result['r2_membrane'],
                        'R2_Cytoskeleton': result['r2_cytoskeleton']
                    })

            if results_list:
                csv_data = pd.DataFrame(results_list).to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col3:
            if drive_enabled and GoogleDriveManager is not None:
                if st.button("☁️ Upload to Google Drive", use_container_width=True):
                    st.info("Google Drive upload: Configure credentials in Streamlit secrets")

    else:
        st.info("👈 Run analysis first to export results")


with tabs[3]:  # About
    st.markdown("""
    ## AFM Cell Batch Analyzer v2.0

    This tool analyzes multiple AFM compression force curves from Igor Pro data files using the Lulevich model.

    ### Features
    - **Batch Processing**: Analyze 5-20 cells in one session
    - **Igor File Support**: Reads native Igor .ibw binary files
    - **Cantilever Editing**: Set cantilever constant per cell or experiment
    - **Baseline Correction**: Auto-detect with manual adjustment options
    - **Force Curve Generation**: Converts raw AFM data to force vs deformation
    - **Lulevich Analysis**: Extracts membrane and cytoskeleton Young's modulus
    - **Comparison Plots**: Visualize results across all cells

    ### Model Reference

    **Membrane Elasticity (Balloon Model, ε < 0.3):**

    F ≈ (2π Em h R₀ ε³) / (1 - νm)

    **Cytoskeleton Elasticity (Hertzian Model):**

    Fi = (√2 Ei R₀^(1/2) ε^(3/2)) / (3(1 - νi²))

    ### References
    Lulevich, V., Zink, T., Chen, H.-Y., Liu, F.-T., & Liu, G.-y. (2006).
    Cell Mechanics Using Atomic Force Microscopy-Based Single-Cell Compression.
    *Langmuir*, 22(19), 8151–8155.
    """)


def analyze_cell(cell_config, cell_height_m, baseline_method, fitting_mode, eps_min, eps_max):
    """Analyze a single cell"""
    try:
        # Parse Igor files
        if not igor_available:
            return {'success': False, 'error': 'Igor parser not available'}

        parser1 = IgorParser(cell_config['file1'])
        parser2 = IgorParser(cell_config['file2'])

        data1 = parser1.parse()
        data2 = parser2.parse()

        if data1['data'] is None or data2['data'] is None:
            return {'success': False, 'error': 'Failed to parse Igor files'}

        # For now, use the first dataset as deflection
        deflection = data1['data']
        z_position = np.linspace(0, len(deflection) * 0.001, len(deflection))  # Approximate Z

        # Baseline correction
        if baseline_available:
            corrector = BaselineCorrector(deflection, z_position)
            baseline_info = corrector.auto_detect_baseline(method='linear' if baseline_method == 'Linear' else 'flat')
            deflection_corrected = corrector.correct_baseline()
            contact_idx = corrector.estimate_contact_point(deflection_corrected)
        else:
            deflection_corrected = deflection - np.mean(deflection[:len(deflection)//10])
            contact_idx = len(deflection) // 2

        # Calculate relative deformation
        rel_def = calculate_relative_deformation(z_position, contact_idx, cell_config['height_m'])

        # Calculate force
        force = deflection_corrected * cell_config['cantilever'] * 1000  # Convert to pN

        # Lulevich analysis
        model = LulevichModel(force, rel_def, cell_config['height_m'])

        if fitting_mode == "Auto Detect":
            auto_range = model.auto_detect_elastic_range()
            eps_min_use = auto_range['elastic_epsilon_min']
            eps_max_use = auto_range['elastic_epsilon_max']
        else:
            eps_min_use = eps_min
            eps_max_use = eps_max

        membrane = model.fit_membrane_elasticity(epsilon_min=eps_min_use, epsilon_max=eps_max_use)
        rupture = model.detect_rupture_point()
        cytoskeleton = model.fit_cytoskeleton_elasticity(epsilon_min=0.05, epsilon_max=min(0.3, rupture['epsilon']*0.9))

        return {
            'success': True,
            'cell_id': cell_config['id'],
            'Em_MPa': membrane.get('Em_MPa', 0),
            'Km_kT': membrane.get('Km_kT', 0),
            'Ei_kPa': cytoskeleton.get('Ei_kPa', 0),
            'r2_membrane': membrane.get('r_squared', 0),
            'r2_cytoskeleton': cytoskeleton.get('r_squared', 0),
            'rupture_epsilon': rupture.get('epsilon', 0)
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}
