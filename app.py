"""
AFM Cell Analyzer v3 - Enhanced Workflow
Supports both Igor files and pre-processed force curves
Cantilever constant optional, Spring constant added
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

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
    .section-header {
        color: #2ca02c;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 1em;
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
if 'data' not in st.session_state:
    st.session_state.data = None
if 'gs_manager' not in st.session_state:
    st.session_state.gs_manager = None
if 'current_cell_id' not in st.session_state:
    st.session_state.current_cell_id = ""
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔬 AFM Cell Analyzer</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**v3.0** - Dual Workflow")

st.markdown("---")

# Sidebar - Settings and Database Connection
with st.sidebar:
    st.markdown("## ⚙️ Settings & Connection")

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
                st.success("✅ Connected to Google Sheets!")
                st.info(f"📊 Database: {gs_manager.get_spreadsheet_url()}")
        else:
            gs_manager = st.session_state.gs_manager

    st.markdown("---")
    st.markdown("### Analysis Settings")

    # Cell height
    cell_height_um = st.number_input(
        "Default Cell Height (μm)",
        min_value=1.0,
        max_value=50.0,
        value=8.09,
        step=0.1,
        help="Can be overridden per cell"
    )

    # Spring constant (ALWAYS needed for Lulevich model)
    spring_constant_nm = st.number_input(
        "Spring Constant (N/m)",
        min_value=0.001,
        max_value=100.0,
        value=0.05,
        step=0.001,
        help="Spring constant for mechanical property calculations"
    )

    st.markdown("---")
    st.markdown("### Baseline Detection")

    baseline_type = st.selectbox(
        "Baseline Type",
        ["Auto-detect (Flat)", "Linear", "Manual"],
        help="How to detect the baseline"
    )

    st.markdown("---")
    st.markdown("### Analysis Range")

    fitting_mode = st.radio(
        "Fitting Method",
        ["Auto Detect", "Manual Range"],
        help="Auto: Automatic range detection | Manual: User-specified ranges"
    )

    if fitting_mode == "Manual Range":
        col1, col2 = st.columns(2)
        with col1:
            eps_min = st.number_input(
                "ε min",
                min_value=0.0,
                max_value=0.5,
                value=0.02,
                step=0.01
            )
        with col2:
            eps_max = st.number_input(
                "ε max",
                min_value=0.05,
                max_value=0.5,
                value=0.3,
                step=0.01
            )
    else:
        eps_min, eps_max = None, None

# Main tabs
tabs = st.tabs([
    "📤 Upload & Analyze",
    "📊 Database Browser",
    "📈 Results & Visualization",
    "💾 Export"
])

# ==================== TAB 1: Upload & Analyze ====================
with tabs[0]:
    st.markdown("## Upload & Analyze Cellular Data")

    # Choose workflow
    workflow_type = st.radio(
        "Select Analysis Workflow",
        ["Igor Binary Files (.ibw)", "Pre-processed Force Curve (CSV/Excel)"],
        horizontal=True,
        help="Igor: Raw AFM data | Force Curve: Pre-calculated force vs. relative deformation"
    )

    st.markdown("---")

    # ========== WORKFLOW A: Igor Files ==========
    if workflow_type == "Igor Binary Files (.ibw)":
        st.markdown("### 📁 Igor File Analysis")
        st.info("Upload raw Igor .ibw files. Cantilever constant will be used to calculate force.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cell Information")

            cell_id = st.text_input(
                "Cell ID/Name *",
                value=st.session_state.current_cell_id,
                placeholder="e.g., C2C12_001",
                help="Required: Unique identifier for this cell"
            )
            st.session_state.current_cell_id = cell_id

            cell_height_upload = st.number_input(
                "Cell Height (μm) *",
                min_value=1.0,
                max_value=50.0,
                value=cell_height_um,
                step=0.1
            )

            cantilever_constant_nm = st.number_input(
                "Cantilever Constant (N/m) *",
                min_value=0.001,
                max_value=1000.0,
                value=0.05,
                step=0.001,
                help="Spring constant of the AFM cantilever (in N/m)"
            )

        with col2:
            st.markdown("### Optional Metadata")

            video_link = st.text_input(
                "Google Drive Video Link (optional)",
                placeholder="https://drive.google.com/file/d/...",
                help="Link to compression video"
            )

            create_force_curve = st.checkbox(
                "Create Force vs RelDef Curve?",
                value=True,
                help="Generate force-deformation plot"
            )

            analysis_notes = st.text_area(
                "Analysis Notes (optional)",
                placeholder="Any observations...",
                height=100
            )

        st.markdown("---")
        st.markdown("### Upload Igor Files")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Igor File 1 (Approach)**")
            igor_file1 = st.file_uploader(
                "Select first Igor file (.ibw)",
                type=['ibw'],
                key="igor1",
                help="Approach or first compression curve"
            )

        with col2:
            st.markdown("**Igor File 2 (Retract)**")
            igor_file2 = st.file_uploader(
                "Select second Igor file (.ibw)",
                type=['ibw'],
                key="igor2",
                help="Retract or second compression curve"
            )

        # File validation and preview
        if igor_file1 is not None and igor_file2 is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### File 1 Preview")
                with st.spinner("Parsing Igor file 1..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.ibw') as tmp1:
                        tmp1.write(igor_file1.read())
                        tmp1_path = tmp1.name

                    try:
                        parser1 = IgorParser(tmp1_path)
                        result1 = parser1.parse()
                        data1 = result1['data']

                        if data1 is not None:
                            st.success(f"✅ Loaded {len(data1)} data points")
                            st.write(f"**Range**: {data1.min():.2e} to {data1.max():.2e}")
                        else:
                            st.error("❌ Could not extract data from file 1")
                            data1 = None

                    except Exception as e:
                        st.error(f"❌ Error parsing file 1: {e}")
                        data1 = None
                    finally:
                        os.unlink(tmp1_path)

            with col2:
                st.markdown("#### File 2 Preview")
                with st.spinner("Parsing Igor file 2..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.ibw') as tmp2:
                        tmp2.write(igor_file2.read())
                        tmp2_path = tmp2.name

                    try:
                        parser2 = IgorParser(tmp2_path)
                        result2 = parser2.parse()
                        data2 = result2['data']

                        if data2 is not None:
                            st.success(f"✅ Loaded {len(data2)} data points")
                            st.write(f"**Range**: {data2.min():.2e} to {data2.max():.2e}")
                        else:
                            st.error("❌ Could not extract data from file 2")
                            data2 = None

                    except Exception as e:
                        st.error(f"❌ Error parsing file 2: {e}")
                        data2 = None
                    finally:
                        os.unlink(tmp2_path)

            if data1 is not None and data2 is not None:
                st.session_state.uploaded_data = (data1, data2)

                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    run_analysis = st.button(
                        "🚀 Analyze Igor Files",
                        use_container_width=True,
                        type="primary"
                    )

                with col2:
                    st.empty()

                with col3:
                    st.empty()

                if run_analysis:
                    if not cell_id:
                        st.error("❌ Cell ID is required")
                    else:
                        with st.spinner("Analyzing..."):
                            try:
                                # Baseline correction
                                baseline_method = "flat" if baseline_type == "Auto-detect (Flat)" else "linear"
                                corrector = BaselineCorrector(data1, np.arange(len(data1)))
                                baseline_info = corrector.auto_detect_baseline(method=baseline_method)
                                deflection_corrected = corrector.correct_baseline()

                                # Calculate force from deflection
                                force = deflection_corrected * cantilever_constant_nm * 1e9  # Convert to pN

                                # Estimate contact point
                                contact_idx = corrector.estimate_contact_point(deflection_corrected)

                                # Calculate relative deformation
                                z_position = np.arange(len(data1))
                                relative_def = calculate_relative_deformation(
                                    z_position,
                                    contact_idx,
                                    cell_height_upload
                                )

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
                                    'cell_id': cell_id,
                                    'Em': fit_results_membrane['Em'],
                                    'Ei': fit_results_cyto['Ei'],
                                    'r2_membrane': fit_results_membrane.get('r2', 0),
                                    'r2_cyto': fit_results_cyto.get('r2', 0),
                                    'force': force,
                                    'relative_def': relative_def,
                                    'cell_height': cell_height_upload,
                                    'cantilever_constant': cantilever_constant_nm,
                                    'spring_constant': spring_constant_nm,
                                    'video_link': video_link,
                                    'force_curve_created': create_force_curve,
                                    'notes': analysis_notes,
                                    'timestamp': datetime.now()
                                }

                                st.success("✅ Analysis Complete!")
                                st.session_state.data = (data1, data2)

                                # Save to database
                                if enable_database and gs_manager:
                                    cell_data = {
                                        'cell_id': cell_id,
                                        'date_analyzed': datetime.now().strftime("%Y-%m-%d"),
                                        'cell_height': cell_height_upload,
                                        'cantilever_constant': cantilever_constant_nm,
                                        'Em': round(fit_results_membrane['Em'], 4),
                                        'Ei': round(fit_results_cyto['Ei'], 4),
                                        'video_link': video_link,
                                        'force_curve_created': 'Yes' if create_force_curve else 'No',
                                        'fit_quality': round(fit_results_membrane.get('r2', 0), 4),
                                        'notes': analysis_notes,
                                        'analysis_status': 'Complete'
                                    }
                                    success, msg = gs_manager.append_cell_data(cell_data)
                                    st.info(msg)

                                # Display results
                                st.markdown("---")
                                st.markdown("### Results")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Em (Membrane)", f"{fit_results_membrane['Em']:.2f} MPa")
                                    st.metric("R² (Membrane)", f"{fit_results_membrane.get('r2', 0):.4f}")

                                with col2:
                                    st.metric("Ei (Cytoskeleton)", f"{fit_results_cyto['Ei']:.2f} kPa")
                                    st.metric("R² (Cytoskeleton)", f"{fit_results_cyto.get('r2', 0):.4f}")

                                # Force curve
                                if create_force_curve:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=relative_def,
                                        y=force / 1e9,
                                        mode='lines+markers',
                                        name='Force vs Relative Deformation'
                                    ))
                                    fig.update_layout(
                                        title="Force vs Relative Deformation",
                                        xaxis_title="Relative Deformation (ε)",
                                        yaxis_title="Force (nN)",
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ Analysis Error: {str(e)}")

    # ========== WORKFLOW B: Pre-processed Force Curve ==========
    else:
        st.markdown("### 📊 Force Curve Analysis")
        st.info("Upload pre-processed force vs. relative deformation data. Spring constant will be used for analysis.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cell Information")

            cell_id = st.text_input(
                "Cell ID/Name *",
                value=st.session_state.current_cell_id,
                placeholder="e.g., C2C12_001",
                help="Required: Unique identifier for this cell"
            )
            st.session_state.current_cell_id = cell_id

            cell_height_upload = st.number_input(
                "Cell Height (μm) *",
                min_value=1.0,
                max_value=50.0,
                value=cell_height_um,
                step=0.1
            )

        with col2:
            st.markdown("### Analysis Parameters")

            spring_constant_input = st.number_input(
                "Spring Constant (N/m) *",
                min_value=0.001,
                max_value=100.0,
                value=spring_constant_nm,
                step=0.001,
                help="Spring constant for mechanical property calculations"
            )

            video_link = st.text_input(
                "Google Drive Video Link (optional)",
                placeholder="https://drive.google.com/file/d/...",
                help="Link to compression video"
            )

        st.markdown("---")
        st.markdown("### Upload Force Curve Data")

        st.markdown("Expected file format: CSV or Excel with columns: **relative_deformation** (ε) and **force** (nN)")

        force_curve_file = st.file_uploader(
            "Upload force curve file (.csv or .xlsx)",
            type=['csv', 'xlsx'],
            key="force_curve_file",
            help="Pre-calculated force vs. relative deformation"
        )

        if force_curve_file is not None:
            try:
                if force_curve_file.name.endswith('.csv'):
                    df = pd.read_csv(force_curve_file)
                else:
                    df = pd.read_excel(force_curve_file)

                st.success(f"✅ Loaded {len(df)} data points")
                st.dataframe(df.head(10))

                # Extract columns
                col_names = df.columns.tolist()
                col1, col2 = st.columns(2)

                with col1:
                    eps_col = st.selectbox(
                        "Select Relative Deformation Column",
                        col_names,
                        help="Column containing ε values"
                    )

                with col2:
                    force_col = st.selectbox(
                        "Select Force Column",
                        col_names,
                        help="Column containing force values (nN)"
                    )

                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    run_analysis = st.button(
                        "🚀 Analyze Force Curve",
                        use_container_width=True,
                        type="primary"
                    )

                with col2:
                    st.empty()

                with col3:
                    st.empty()

                if run_analysis:
                    if not cell_id:
                        st.error("❌ Cell ID is required")
                    else:
                        with st.spinner("Analyzing..."):
                            try:
                                # Extract data
                                relative_def = df[eps_col].values
                                force = df[force_col].values

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
                                    'cell_id': cell_id,
                                    'Em': fit_results_membrane['Em'],
                                    'Ei': fit_results_cyto['Ei'],
                                    'r2_membrane': fit_results_membrane.get('r2', 0),
                                    'r2_cyto': fit_results_cyto.get('r2', 0),
                                    'force': force,
                                    'relative_def': relative_def,
                                    'cell_height': cell_height_upload,
                                    'spring_constant': spring_constant_input,
                                    'video_link': video_link,
                                    'force_curve_created': True,
                                    'timestamp': datetime.now()
                                }

                                st.success("✅ Analysis Complete!")

                                # Save to database
                                if enable_database and gs_manager:
                                    cell_data = {
                                        'cell_id': cell_id,
                                        'date_analyzed': datetime.now().strftime("%Y-%m-%d"),
                                        'cell_height': cell_height_upload,
                                        'cantilever_constant': 'N/A (Pre-processed)',
                                        'Em': round(fit_results_membrane['Em'], 4),
                                        'Ei': round(fit_results_cyto['Ei'], 4),
                                        'video_link': video_link,
                                        'force_curve_created': 'Yes',
                                        'fit_quality': round(fit_results_membrane.get('r2', 0), 4),
                                        'notes': 'Pre-processed force curve',
                                        'analysis_status': 'Complete'
                                    }
                                    success, msg = gs_manager.append_cell_data(cell_data)
                                    st.info(msg)

                                # Display results
                                st.markdown("---")
                                st.markdown("### Results")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Em (Membrane)", f"{fit_results_membrane['Em']:.2f} MPa")
                                    st.metric("R² (Membrane)", f"{fit_results_membrane.get('r2', 0):.4f}")

                                with col2:
                                    st.metric("Ei (Cytoskeleton)", f"{fit_results_cyto['Ei']:.2f} kPa")
                                    st.metric("R² (Cytoskeleton)", f"{fit_results_cyto.get('r2', 0):.4f}")

                                # Force curve
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=relative_def,
                                    y=force,
                                    mode='lines+markers',
                                    name='Force vs Relative Deformation'
                                ))
                                fig.update_layout(
                                    title="Force vs Relative Deformation",
                                    xaxis_title="Relative Deformation (ε)",
                                    yaxis_title="Force (nN)",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ Analysis Error: {str(e)}")

            except Exception as e:
                st.error(f"❌ File Error: {str(e)}")

# ==================== TAB 2: Database Browser ====================
with tabs[1]:
    st.markdown("## Database Browser")

    if gs_manager is None and enable_database:
        st.warning("⚠️ Database not connected. Enable in sidebar and click Connect.")
    elif enable_database and gs_manager:
        df_all = gs_manager.get_all_cells()

        if df_all.empty:
            st.info("📋 No cells in database yet.")
        else:
            st.success(f"✅ Loaded {len(df_all)} cells from database")

            # Search and filter
            col1, col2, col3 = st.columns(3)

            with col1:
                search_term = st.text_input(
                    "Search Cell ID",
                    placeholder="e.g., Sample_001"
                )

            with col2:
                sort_by = st.selectbox(
                    "Sort By",
                    ["Cell ID", "Date", "Em (MPa)"]
                )

            with col3:
                show_count = st.number_input(
                    "Show Records",
                    min_value=5,
                    max_value=len(df_all),
                    value=min(20, len(df_all))
                )

            # Apply search
            if search_term:
                df_display = gs_manager.search_cells(search_term)
            else:
                df_display = df_all.copy()

            # Apply sort
            if sort_by == "Em (MPa)" and "Young's Modulus (Em, MPa)" in df_display.columns:
                df_display = df_display.sort_values("Young's Modulus (Em, MPa)", ascending=False)
            elif sort_by == "Date" and "Date Analyzed" in df_display.columns:
                df_display = df_display.sort_values("Date Analyzed", ascending=False)

            # Display
            st.dataframe(df_display.head(show_count), use_container_width=True)

            # Statistics
            st.markdown("---")
            st.markdown("### Statistics")

            stats = gs_manager.get_statistics()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Cells", stats.get('total_cells', 0))

            with col2:
                em_avg = stats.get('avg_em', 0)
                st.metric("Avg Em", f"{em_avg:.2f} MPa" if em_avg > 0 else "N/A")

            with col3:
                ei_avg = stats.get('avg_ei', 0)
                st.metric("Avg Ei", f"{ei_avg:.2f} kPa" if ei_avg > 0 else "N/A")

            with col4:
                st.metric("With Videos", stats.get('cells_with_video', 0))
    else:
        st.info("💡 Enable database in sidebar to view stored cells.")

# ==================== TAB 3: Results & Visualization ====================
with tabs[2]:
    st.markdown("## Results & Visualization")

    if st.session_state.results is None:
        st.info("📊 No analysis results yet. Complete an analysis in Tab 1.")
    else:
        results = st.session_state.results

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Mechanical Properties")
            st.metric("Em (Membrane)", f"{results['Em']:.2f} MPa")
            st.metric("R² (Membrane)", f"{results['r2_membrane']:.4f}")

        with col2:
            st.markdown("### Cytoskeleton Properties")
            st.metric("Ei (Cytoskeleton)", f"{results['Ei']:.2f} kPa")
            st.metric("R² (Cytoskeleton)", f"{results['r2_cyto']:.4f}")

        st.markdown("---")

        # Force vs Relative Deformation
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
            yaxis_title="Force (pN)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: Export ====================
with tabs[3]:
    st.markdown("## Export Data")

    if gs_manager is None or not enable_database:
        st.warning("⚠️ Database not enabled. Cannot export.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Export as CSV"):
                csv_data = gs_manager.export_to_csv()
                if csv_data:
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="afm_cells.csv",
                        mime="text/csv"
                    )

        with col2:
            if st.button("📥 Export as JSON"):
                json_data = gs_manager.export_to_json()
                if json_data:
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="afm_cells.json",
                        mime="application/json"
                    )

        with col3:
            st.info("💡 Excel export coming soon")

        st.markdown("---")
        st.success("✅ Exports include all analyzed cells with complete metadata")
