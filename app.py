"""
AFM Cell Analyzer v4 - Force Curve Focused
Primary: Upload pre-processed force curves
Secondary: Generate force curves from Igor files
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
from datetime import datetime

from lulevich_model import LulevichModel
from igor_parser import IgorParser
from baseline_correction import BaselineCorrector, calculate_relative_deformation
from google_sheets_manager import GoogleSheetsManager, initialize_sheets_manager

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
    .success-box {
        background-color: #d4edda;
        padding: 1em;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1em;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1em;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'gs_manager' not in st.session_state:
    st.session_state.gs_manager = None
if 'generated_force_curve' not in st.session_state:
    st.session_state.generated_force_curve = None

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔬 AFM Cell Analyzer</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**v4.0** - Force Curve Analysis")

st.markdown("---")

# Sidebar - Settings and Database Connection
with st.sidebar:
    st.markdown("## ⚙️ Settings & Database")

    # Google Sheets connection
    st.markdown("### Google Sheets Database")

    enable_database = st.checkbox(
        "Enable Google Sheets Database",
        value=False,
        help="Store analysis results in Google Sheets"
    )

    gs_manager = None
    if enable_database:
        if st.button("🔗 Connect to Google Sheets"):
            gs_manager = initialize_sheets_manager()
            if gs_manager:
                st.session_state.gs_manager = gs_manager
                st.success("✅ Connected!")
        else:
            gs_manager = st.session_state.gs_manager

    st.markdown("---")
    st.markdown("### Analysis Settings")

    # Cell height (for context only, not always needed)
    cell_height_default = st.number_input(
        "Default Cell Height (μm)",
        min_value=1.0,
        max_value=50.0,
        value=8.09,
        step=0.1,
        help="Reference cell height"
    )

    # Spring constant (OPTIONAL)
    spring_constant_default = st.number_input(
        "Default Spring Constant (N/m)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.001,
        help="Optional: Spring constant for model (0 = not used)"
    )

    st.markdown("---")
    st.markdown("### Fitting Options")

    fitting_mode = st.radio(
        "Fitting Method",
        ["Auto Detect", "Manual Range"],
        help="Auto: Automatic range | Manual: User-specified"
    )

    if fitting_mode == "Manual Range":
        col1, col2 = st.columns(2)
        with col1:
            eps_min = st.number_input("ε min", min_value=0.0, max_value=0.5, value=0.02, step=0.01)
        with col2:
            eps_max = st.number_input("ε max", min_value=0.05, max_value=0.5, value=0.3, step=0.01)
    else:
        eps_min, eps_max = None, None

# Main tabs
tabs = st.tabs([
    "📊 Analyze Force Curve",
    "🔧 Generate Force Curve (Igor)",
    "📋 Database Browser",
    "📈 Results",
    "💾 Export"
])

# ==================== TAB 1: Analyze Force Curve ====================
with tabs[0]:
    st.markdown("## Analyze Force vs Relative Deformation Curve")

    st.info("Upload a pre-processed force vs. relative deformation file (CSV or Excel)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cell Information")

        cell_name = st.text_input(
            "Cell Name/ID *",
            placeholder="e.g., C2C12_001",
            help="Required: Unique identifier for this cell"
        )

        date_acquired = st.date_input(
            "Date Acquired *",
            value=datetime.now().date(),
            help="Date when the measurement was taken"
        )

    with col2:
        st.markdown("### Analysis Metadata")

        video_link = st.text_input(
            "Google Drive Video Link (optional)",
            placeholder="https://drive.google.com/file/d/...",
            help="Link to compression video"
        )

        spring_constant = st.number_input(
            "Spring Constant (N/m) (optional)",
            min_value=0.0,
            max_value=100.0,
            value=spring_constant_default,
            step=0.001,
            help="Spring constant for model (0 = not used)"
        )

    st.markdown("---")
    st.markdown("### Upload Force Curve File")

    st.markdown("**Expected format:** CSV or Excel with columns for:")
    st.markdown("- **Relative Deformation (ε)** - dimensionless compression")
    st.markdown("- **Force (nN)** - force in nanonewtons")

    force_curve_file = st.file_uploader(
        "Select force curve file (.csv or .xlsx)",
        type=['csv', 'xlsx'],
        key="force_curve_file"
    )

    if force_curve_file is not None:
        try:
            # Load file
            if force_curve_file.name.endswith('.csv'):
                df = pd.read_csv(force_curve_file)
            else:
                df = pd.read_excel(force_curve_file)

            st.success(f"✅ Loaded {len(df)} data points")
            st.dataframe(df.head(10), use_container_width=True)

            # Column selection
            col_names = df.columns.tolist()
            col1, col2 = st.columns(2)

            with col1:
                eps_col = st.selectbox(
                    "Relative Deformation Column",
                    col_names,
                    help="Column containing ε values"
                )

            with col2:
                force_col = st.selectbox(
                    "Force Column",
                    col_names,
                    help="Column containing force values (nN)"
                )

            st.markdown("---")

            if st.button("🚀 Analyze Force Curve", type="primary", use_container_width=True):
                if not cell_name:
                    st.error("❌ Cell Name is required")
                else:
                    with st.spinner("Analyzing..."):
                        try:
                            # Extract data
                            relative_def = df[eps_col].values.astype(float)
                            force = df[force_col].values.astype(float)

                            # Lulevich model fitting
                            model = LulevichModel(force, relative_def)

                            if fitting_mode == "Manual Range":
                                fit_results_membrane = model.fit_membrane_elasticity(
                                    epsilon_min=eps_min,
                                    epsilon_max=eps_max
                                )
                                fit_results_cyto = model.fit_cytoskeleton_elasticity(
                                    epsilon_min=eps_min,
                                    epsilon_max=eps_max
                                )
                            else:
                                auto_range = model.auto_detect_elastic_range()
                                fit_results_membrane = model.fit_membrane_elasticity(
                                    epsilon_min=auto_range['epsilon_min'],
                                    epsilon_max=auto_range['epsilon_max']
                                )
                                fit_results_cyto = model.fit_cytoskeleton_elasticity(
                                    epsilon_min=auto_range['epsilon_min'],
                                    epsilon_max=auto_range['epsilon_max']
                                )

                            # Store results
                            st.session_state.results = {
                                'cell_name': cell_name,
                                'date_acquired': str(date_acquired),
                                'Em': fit_results_membrane['Em'],
                                'Ei': fit_results_cyto['Ei'],
                                'r2_membrane': fit_results_membrane.get('r2', 0),
                                'r2_cyto': fit_results_cyto.get('r2', 0),
                                'force': force,
                                'relative_def': relative_def,
                                'spring_constant': spring_constant,
                                'video_link': video_link,
                                'timestamp': datetime.now()
                            }

                            st.success("✅ Analysis Complete!")

                            # Save to database
                            if enable_database and gs_manager:
                                cell_data = {
                                    'cell_id': cell_name,
                                    'date_analyzed': date_acquired.strftime("%Y-%m-%d"),
                                    'cell_height': 'N/A',
                                    'cantilever_constant': 'N/A',
                                    'Em': round(fit_results_membrane['Em'], 4),
                                    'Ei': round(fit_results_cyto['Ei'], 4),
                                    'video_link': video_link,
                                    'force_curve_created': 'Yes',
                                    'fit_quality': round(fit_results_membrane.get('r2', 0), 4),
                                    'notes': f'Spring constant: {spring_constant} N/m',
                                    'analysis_status': 'Complete'
                                }
                                success, msg = gs_manager.append_cell_data(cell_data)
                                st.info(msg)

                            # Display results
                            st.markdown("---")
                            st.markdown("### Analysis Results")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Em (Membrane)", f"{fit_results_membrane['Em']:.2f} MPa")
                                st.metric("R² (Membrane)", f"{fit_results_membrane.get('r2', 0):.4f}")

                            with col2:
                                st.metric("Ei (Cytoskeleton)", f"{fit_results_cyto['Ei']:.2f} kPa")
                                st.metric("R² (Cytoskeleton)", f"{fit_results_cyto.get('r2', 0):.4f}")

                            # Force curve plot
                            st.markdown("---")
                            st.markdown("### Force vs Relative Deformation")

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=relative_def,
                                y=force,
                                mode='lines+markers',
                                name='Measurement',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=5)
                            ))
                            fig.update_layout(
                                title="Force vs Relative Deformation",
                                xaxis_title="Relative Deformation (ε)",
                                yaxis_title="Force (nN)",
                                hovermode='x unified',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Analysis Error: {str(e)}")

        except Exception as e:
            st.error(f"❌ File Error: {str(e)}")

# ==================== TAB 2: Generate Force Curve from Igor ====================
with tabs[1]:
    st.markdown("## Generate Force Curve from Igor Files")

    st.info("Create a force vs. relative deformation file from Igor binary data. Upload two files: one from the surface (baseline) and one from the cell (compression).")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### File 1: Surface Reference")
        igor_surface = st.file_uploader(
            "Select surface Igor file (.ibw)",
            type=['ibw'],
            key="igor_surface",
            help="Surface/baseline measurement for reference"
        )

    with col2:
        st.markdown("### File 2: Cell Compression")
        igor_cell = st.file_uploader(
            "Select cell Igor file (.ibw)",
            type=['ibw'],
            key="igor_cell",
            help="Cell compression measurement"
        )

    if igor_surface is not None and igor_cell is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Surface File Preview")
            with st.spinner("Parsing surface file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ibw') as tmp:
                    tmp.write(igor_surface.read())
                    tmp_path = tmp.name

                try:
                    parser = IgorParser(tmp_path)
                    result = parser.parse()
                    data_surface = result['data']

                    if data_surface is not None:
                        st.success(f"✅ Loaded {len(data_surface)} points")
                        st.write(f"Range: {data_surface.min():.2e} to {data_surface.max():.2e}")
                    else:
                        st.error("❌ Could not extract data")
                        data_surface = None
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    data_surface = None
                finally:
                    os.unlink(tmp_path)

        with col2:
            st.markdown("#### Cell File Preview")
            with st.spinner("Parsing cell file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ibw') as tmp:
                    tmp.write(igor_cell.read())
                    tmp_path = tmp.name

                try:
                    parser = IgorParser(tmp_path)
                    result = parser.parse()
                    data_cell = result['data']

                    if data_cell is not None:
                        st.success(f"✅ Loaded {len(data_cell)} points")
                        st.write(f"Range: {data_cell.min():.2e} to {data_cell.max():.2e}")
                    else:
                        st.error("❌ Could not extract data")
                        data_cell = None
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    data_cell = None
                finally:
                    os.unlink(tmp_path)

        if data_surface is not None and data_cell is not None:
            st.markdown("---")
            st.markdown("### Generation Parameters")

            col1, col2 = st.columns(2)

            with col1:
                cantilever_constant = st.number_input(
                    "Cantilever Constant (N/m)",
                    min_value=0.001,
                    max_value=1000.0,
                    value=0.05,
                    step=0.001,
                    help="Spring constant of AFM cantilever"
                )

            with col2:
                cell_height = st.number_input(
                    "Cell Height (μm)",
                    min_value=1.0,
                    max_value=50.0,
                    value=cell_height_default,
                    step=0.1
                )

            st.markdown("---")

            if st.button("⚙️ Generate Force Curve", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        # Use cell data for analysis
                        baseline_corrector = BaselineCorrector(data_cell, np.arange(len(data_cell)))
                        baseline_info = baseline_corrector.auto_detect_baseline(method='flat')
                        deflection_corrected = baseline_corrector.correct_baseline()

                        # Calculate force
                        force = deflection_corrected * cantilever_constant * 1e9  # Convert to pN

                        # Estimate contact point
                        contact_idx = baseline_corrector.estimate_contact_point(deflection_corrected)

                        # Calculate relative deformation
                        z_position = np.arange(len(data_cell))
                        relative_def = calculate_relative_deformation(z_position, contact_idx, cell_height)

                        # Store generated curve
                        st.session_state.generated_force_curve = {
                            'relative_def': relative_def,
                            'force': force
                        }

                        st.success("✅ Force Curve Generated!")

                        # Create download file
                        df_output = pd.DataFrame({
                            'Relative Deformation': relative_def,
                            'Force (nN)': force / 1e3  # Convert pN to nN
                        })

                        csv = df_output.to_csv(index=False)

                        st.download_button(
                            label="📥 Download Force Curve (CSV)",
                            data=csv,
                            file_name="force_curve.csv",
                            mime="text/csv"
                        )

                        # Preview
                        st.markdown("---")
                        st.markdown("### Generated Data Preview")
                        st.dataframe(df_output.head(20), use_container_width=True)

                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=relative_def,
                            y=force / 1e3,
                            mode='lines+markers',
                            name='Generated Force Curve'
                        ))
                        fig.update_layout(
                            title="Generated Force vs Relative Deformation",
                            xaxis_title="Relative Deformation (ε)",
                            yaxis_title="Force (nN)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.info("💡 Download the CSV file and upload it to Tab 1 to analyze")

                    except Exception as e:
                        st.error(f"❌ Generation Error: {str(e)}")

# ==================== TAB 3: Database Browser ====================
with tabs[2]:
    st.markdown("## Database Browser")

    if gs_manager is None or not enable_database:
        st.warning("⚠️ Database not connected. Enable in sidebar.")
    else:
        df_all = gs_manager.get_all_cells()

        if df_all.empty:
            st.info("📋 No cells in database yet.")
        else:
            st.success(f"✅ {len(df_all)} cells in database")

            col1, col2 = st.columns(2)

            with col1:
                search_term = st.text_input("Search Cell ID", placeholder="e.g., C2C12_001")

            with col2:
                show_count = st.number_input("Show Records", min_value=5, max_value=len(df_all), value=min(20, len(df_all)))

            if search_term:
                df_display = gs_manager.search_cells(search_term)
            else:
                df_display = df_all.copy()

            st.dataframe(df_display.head(show_count), use_container_width=True)

            # Statistics
            st.markdown("---")
            stats = gs_manager.get_statistics()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cells", stats.get('total_cells', 0))
            with col2:
                em = stats.get('avg_em', 0)
                st.metric("Avg Em", f"{em:.2f} MPa" if em > 0 else "N/A")
            with col3:
                ei = stats.get('avg_ei', 0)
                st.metric("Avg Ei", f"{ei:.2f} kPa" if ei > 0 else "N/A")
            with col4:
                st.metric("With Videos", stats.get('cells_with_video', 0))

# ==================== TAB 4: Results ====================
with tabs[3]:
    st.markdown("## Analysis Results")

    if st.session_state.results is None:
        st.info("📊 No results yet. Complete an analysis in Tab 1.")
    else:
        results = st.session_state.results

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cell & Acquisition")
            st.write(f"**Cell:** {results['cell_name']}")
            st.write(f"**Date:** {results['date_acquired']}")

        with col2:
            st.markdown("### Mechanical Properties")
            st.metric("Em (Membrane)", f"{results['Em']:.2f} MPa")
            st.metric("Ei (Cytoskeleton)", f"{results['Ei']:.2f} kPa")

        st.markdown("---")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['relative_def'],
            y=results['force'],
            mode='lines+markers',
            name='Force vs Relative Deformation'
        ))
        fig.update_layout(
            title="Force vs Relative Deformation",
            xaxis_title="Relative Deformation (ε)",
            yaxis_title="Force (nN)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: Export ====================
with tabs[4]:
    st.markdown("## Export Data")

    if gs_manager is None or not enable_database:
        st.warning("⚠️ Database not enabled.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export as CSV", use_container_width=True):
                csv_data = gs_manager.export_to_csv()
                if csv_data:
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"afm_cells_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

        with col2:
            if st.button("📥 Export as JSON", use_container_width=True):
                json_data = gs_manager.export_to_json()
                if json_data:
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"afm_cells_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

        st.markdown("---")
        st.success("✅ Exports include all cells with complete metadata")
