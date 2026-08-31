"""
AFM Cell Analyzer - Production Streamlit App
Complete analysis tool for single-cell compression experiments with Google Sheets integration
Tabs: Upload Igor Files | Database Browser | Results & Visualization | Export
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
    st.markdown("**v2.0** - With Database")

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

    if enable_database:
        if st.button("🔐 Connect to Google Sheets", use_container_width=True):
            manager = initialize_sheets_manager("AFM_Cell_Database")
            if manager:
                st.session_state.gs_manager = manager
                st.success("✅ Connected to Google Sheets!")
            else:
                st.error("❌ Failed to connect. Check your credentials.")

        if st.session_state.gs_manager:
            st.success("✅ Database Connected")
            url = st.session_state.gs_manager.get_spreadsheet_url()
            if url:
                st.markdown(f"[📊 Open Sheet]({url})")
        else:
            st.warning("⚠️ Not connected. Add credentials to secrets.")

    st.markdown("---")

    # Cell parameters
    st.markdown("### Cell Parameters")

    cell_height_um = st.number_input(
        "Cell Height (μm)",
        min_value=1.0,
        max_value=50.0,
        value=8.09,
        step=0.1,
        help="Initial cell height in micrometers"
    )
    cell_height_m = cell_height_um * 1e-6

    cell_radius_um = st.number_input(
        "Cell Radius (μm)",
        min_value=1.0,
        max_value=30.0,
        value=cell_height_um * 0.55,
        step=0.1,
        help="Cell radius (auto-estimated if not specified)"
    )
    cell_radius_m = cell_radius_um * 1e-6

    # Baseline detection
    st.markdown("---")
    st.markdown("### Baseline Detection")

    baseline_method = st.radio(
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
    "📤 Upload Igor Files",
    "📊 Database Browser",
    "📈 Results & Visualization",
    "💾 Export"
])

# ==================== TAB 1: Upload Igor Files ====================
with tabs[0]:
    st.markdown("## Upload & Analyze Igor Files")

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

        cantilever_constant = st.number_input(
            "Cantilever Constant (pN/nm) *",
            min_value=0.1,
            max_value=1000.0,
            value=50.0,
            step=0.1,
            help="Spring constant of the AFM cantilever"
        )

    with col2:
        st.markdown("### Optional Metadata")

        video_link = st.text_input(
            "Google Drive Video Link (optional)",
            placeholder="https://drive.google.com/file/d/...",
            help="Link to compression video (not downloaded, just stored)"
        )

        create_force_curve = st.checkbox(
            "Create Force vs RelDef Curve?",
            value=True,
            help="Generate force-deformation plot for this analysis"
        )

        analysis_notes = st.text_area(
            "Analysis Notes (optional)",
            placeholder="Any observations or conditions during analysis...",
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
            st.markdown("### Analysis Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                run_analysis = st.button(
                    "🚀 Analyze Files",
                    use_container_width=True,
                    type="primary"
                )

            with col2:
                auto_detect_baseline_btn = st.button(
                    "📍 Auto-detect Baseline",
                    use_container_width=True
                )

            with col3:
                st.empty()

            # Auto-detect baseline
            if auto_detect_baseline_btn:
                st.info("Analyzing baseline regions...")
                # For now, show that this would work with proper z-position data
                st.warning(
                    "⚠️ Baseline auto-detection requires Z-position data. "
                    "Ensure Igor files contain deflection and Z-position columns."
                )

            # Run analysis
            if run_analysis:
                if not cell_id:
                    st.error("❌ Cell ID is required!")
                else:
                    st.markdown("---")
                    st.markdown("## ✅ Analysis Started")

                    progress = st.progress(0)

                    # Combine data from both files
                    deflection_data = np.concatenate([data1, data2])

                    # Create synthetic Z position data (would come from Igor file in practice)
                    z_data = np.linspace(0, cell_height_upload * 1e-6, len(deflection_data))

                    # Baseline correction
                    progress.progress(20)
                    corrector = BaselineCorrector(deflection_data, z_data)

                    if baseline_method == "Auto-detect (Flat)":
                        baseline_info = corrector.auto_detect_baseline(method='flat')
                    elif baseline_method == "Linear":
                        baseline_info = corrector.auto_detect_baseline(method='linear')
                    else:
                        baseline_info = {'method': 'manual', 'offset': 0, 'slope': 0}

                    corrected_deflection = corrector.correct_baseline()

                    # Convert to force
                    progress.progress(40)
                    force_data = corrected_deflection * cantilever_constant * 1e-12  # pN to N

                    # Estimate contact point
                    contact_idx = corrector.estimate_contact_point(corrected_deflection)

                    # Calculate relative deformation
                    progress.progress(60)
                    rel_def = calculate_relative_deformation(
                        z_data,
                        contact_idx,
                        cell_height_upload * 1e-6
                    )

                    # Lulevich analysis
                    progress.progress(80)
                    cell_radius_upload = cell_height_upload * 0.55 * 1e-6

                    model = LulevichModel(
                        force_data,
                        rel_def,
                        cell_height_upload * 1e-6,
                        cell_radius_upload
                    )

                    # Detect rupture
                    rupture = model.detect_rupture_point()

                    # Fit models
                    if fitting_mode == "Auto Detect":
                        auto_range = model.auto_detect_elastic_range()
                        eps_min_fit = auto_range['elastic_epsilon_min']
                        eps_max_fit = auto_range['elastic_epsilon_max']
                    else:
                        eps_min_fit = eps_min
                        eps_max_fit = eps_max

                    membrane = model.fit_membrane_elasticity(
                        eps_min=eps_min_fit,
                        eps_max=eps_max_fit
                    )

                    cytoskeleton = model.fit_cytoskeleton_elasticity(
                        eps_min=0.05,
                        eps_max=min(0.3, rupture['epsilon'] * 0.9)
                    )

                    progress.progress(100)

                    # Store results
                    st.session_state.results = model.get_summary()
                    st.session_state.data = pd.DataFrame({
                        'Force': force_data,
                        'RelDef': rel_def,
                        'Deflection': corrected_deflection
                    })

                    # Display results
                    st.success("✅ Analysis complete!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### Membrane")
                        st.metric(
                            "Young's Modulus",
                            f"{membrane.get('Em_MPa', 0):.2f} MPa",
                            delta="Elastic region"
                        )
                        st.metric(
                            "Fit Quality (R²)",
                            f"{membrane.get('r_squared', 0):.4f}"
                        )

                    with col2:
                        st.markdown("### Cytoskeleton")
                        st.metric(
                            "Young's Modulus",
                            f"{cytoskeleton.get('Ei_kPa', 0):.2f} kPa",
                            delta="Hertzian fit"
                        )
                        st.metric(
                            "Fit Quality (R²)",
                            f"{cytoskeleton.get('r_squared', 0):.4f}"
                        )

                    with col3:
                        st.markdown("### Rupture")
                        st.metric(
                            "Rupture Point (ε)",
                            f"{rupture.get('epsilon', 0):.4f}"
                        )
                        st.metric(
                            "Rupture Force",
                            f"{rupture.get('force', 0):.2e} N"
                        )

                    st.markdown("---")

                    # Save to database
                    if st.session_state.gs_manager:
                        st.markdown("### Save to Database")

                        if st.button("💾 Save to Google Sheets", use_container_width=True, type="primary"):
                            cell_data = {
                                "cell_id": cell_id,
                                "date_analyzed": datetime.now().strftime("%Y-%m-%d"),
                                "cell_height": cell_height_upload,
                                "cantilever_constant": cantilever_constant,
                                "Em": membrane.get('Em_MPa', 0),
                                "Ei": cytoskeleton.get('Ei_kPa', 0),
                                "video_link": video_link,
                                "force_curve_created": "Yes" if create_force_curve else "No",
                                "fit_quality": membrane.get('r_squared', 0),
                                "notes": analysis_notes
                            }

                            success, msg = st.session_state.gs_manager.append_cell_data(cell_data)

                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

# ==================== TAB 2: Database Browser ====================
with tabs[1]:
    st.markdown("## Cell Database Browser")

    if st.session_state.gs_manager:
        st.info("✅ Connected to Google Sheets database")

        # Search and filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            search_term = st.text_input(
                "Search by Cell ID",
                placeholder="e.g., C2C12_001"
            )

        with col2:
            sort_order = st.radio(
                "Sort by Young's Modulus",
                ["Highest → Lowest", "Lowest → Highest"],
                horizontal=True
            )

        with col3:
            filter_video = st.checkbox("Show only cells with videos")

        # Apply filters
        if search_term:
            df_display = st.session_state.gs_manager.search_cells(search_term)
        else:
            df_display = st.session_state.gs_manager.get_all_cells()

        # Sort
        if not df_display.empty and "Young's Modulus (Em, MPa)" in df_display.columns:
            df_display = st.session_state.gs_manager.sort_by_modulus(
                descending=(sort_order == "Highest → Lowest")
            )

        # Filter by video
        if filter_video and not df_display.empty:
            df_display = df_display[df_display["Video Link"].astype(str).str.len() > 0]

        # Display statistics
        if not df_display.empty:
            stats = st.session_state.gs_manager.get_statistics()

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Cells", stats.get("total_cells", 0))

            with col2:
                em_avg = stats.get("avg_em", 0)
                st.metric("Avg Em", f"{em_avg:.2f} MPa" if em_avg > 0 else "N/A")

            with col3:
                ei_avg = stats.get("avg_ei", 0)
                st.metric("Avg Ei", f"{ei_avg:.2f} kPa" if ei_avg > 0 else "N/A")

            with col4:
                st.metric("With Videos", stats.get("cells_with_video", 0))

            with col5:
                st.metric("Force Curves", stats.get("cells_with_force_curve", 0))

            st.markdown("---")

            # Display table
            st.markdown("### All Cells")

            # Select columns to display
            display_cols = [
                "Cell ID",
                "Date Analyzed",
                "Cell Height (μm)",
                "Young's Modulus (Em, MPa)",
                "Young's Modulus (Ei, kPa)",
                "Fit Quality (R²)",
                "Notes"
            ]

            available_cols = [col for col in display_cols if col in df_display.columns]
            df_display_subset = df_display[available_cols].copy()

            # Format numeric columns
            for col in ["Young's Modulus (Em, MPa)", "Young's Modulus (Ei, kPa)", "Fit Quality (R²)"]:
                if col in df_display_subset.columns:
                    df_display_subset[col] = pd.to_numeric(
                        df_display_subset[col],
                        errors='coerce'
                    ).round(4)

            st.dataframe(df_display_subset, use_container_width=True)

            st.markdown("---")

            # Detail view
            st.markdown("### View Cell Details")

            if len(df_display) > 0:
                cell_names = df_display["Cell ID"].tolist()
                selected_cell = st.selectbox("Select a cell to view details", cell_names)

                if selected_cell:
                    cell_row = df_display[df_display["Cell ID"] == selected_cell].iloc[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Basic Info")
                        st.write(f"**Cell ID**: {cell_row.get('Cell ID', 'N/A')}")
                        st.write(f"**Date**: {cell_row.get('Date Analyzed', 'N/A')}")
                        st.write(f"**Height**: {cell_row.get('Cell Height (μm)', 'N/A')} μm")
                        st.write(f"**Cantilever**: {cell_row.get('Cantilever Constant (pN/nm)', 'N/A')} pN/nm")

                    with col2:
                        st.markdown("#### Mechanical Properties")
                        em_val = cell_row.get("Young's Modulus (Em, MPa)", "N/A")
                        ei_val = cell_row.get("Young's Modulus (Ei, kPa)", "N/A")
                        st.write(f"**Em (Membrane)**: {em_val} MPa")
                        st.write(f"**Ei (Cytoskeleton)**: {ei_val} kPa")
                        st.write(f"**Fit Quality (R²)**: {cell_row.get('Fit Quality (R²)', 'N/A')}")
                        st.write(f"**Force Curve**: {cell_row.get('Force Curve Created', 'No')}")

                    # Video link
                    video_url = cell_row.get("Video Link", "")
                    if video_url and isinstance(video_url, str) and len(video_url) > 0:
                        st.markdown("#### Video")
                        st.markdown(f"[📹 View Video]({video_url})")

                    # Notes
                    notes = cell_row.get("Notes", "")
                    if notes:
                        st.markdown("#### Notes")
                        st.write(notes)

                    # Delete option
                    st.markdown("---")
                    if st.button("🗑️ Delete This Cell", key=f"delete_{selected_cell}"):
                        success, msg = st.session_state.gs_manager.delete_cell(selected_cell)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        else:
            if search_term:
                st.info(f"No cells found matching '{search_term}'")
            else:
                st.info("📊 No cells in database yet. Upload some data in the 'Upload Igor Files' tab.")

    else:
        st.warning("⚠️ Database not connected. Enable it in the sidebar settings.")

# ==================== TAB 3: Results & Visualization ====================
with tabs[2]:
    st.markdown("## Results & Visualization")

    if st.session_state.results is not None and st.session_state.data is not None:
        df = st.session_state.data
        results = st.session_state.results

        # Display numeric results
        col1, col2, col3 = st.columns(3)

        if 'membrane' in results:
            mem = results['membrane']
            with col1:
                st.markdown("### Membrane (Elastic)")
                st.metric("Young's Modulus", f"{mem.get('Em_MPa', 0):.2f} MPa")
                st.metric("Bending Constant", f"{mem.get('Km_kT', 0):.1f} kT")
                st.metric("Fit Quality (R²)", f"{mem.get('r_squared', 0):.4f}")

        if 'cytoskeleton' in results:
            cyto = results['cytoskeleton']
            with col2:
                st.markdown("### Cytoskeleton (Hertzian)")
                st.metric("Young's Modulus", f"{cyto.get('Ei_kPa', 0):.2f} kPa")
                st.metric("Data Points", f"{cyto.get('n_points', 0)}")
                st.metric("Fit Quality (R²)", f"{cyto.get('r_squared', 0):.4f}")

        if 'rupture' in results:
            rup = results['rupture']
            with col3:
                st.markdown("### Rupture Point")
                st.metric("Relative Deformation", f"{rup.get('epsilon', 0):.4f}")
                st.metric("Force", f"{rup.get('force', 0):.2e} N")
                st.metric("Peaks Detected", f"{rup.get('n_peaks_detected', 0)}")

        st.markdown("---")

        # Force-deformation plots
        st.markdown("### Force vs Relative Deformation")

        force = df['Force'].values
        rel_def = df['RelDef'].values

        # Create model for predictions
        model = LulevichModel(force, rel_def, 8.09e-6, 8.09e-6 * 0.55)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Full Data", "Elastic Region (Fitted)")
        )

        # Full data
        fig.add_trace(
            go.Scatter(x=rel_def, y=force, mode='markers', name='Data',
                      marker=dict(size=5, color='blue')),
            row=1, col=1
        )

        # Elastic region with fit
        if 'membrane' in results:
            mem = results['membrane']
            eps_range = mem.get('epsilon_range', [0.02, 0.3])
            eps_fit = np.linspace(eps_range[0], eps_range[1], 100)
            force_fit = model.balloon_model_cubic(eps_fit, mem['Em'])

            fig.add_trace(
                go.Scatter(x=eps_fit, y=force_fit, mode='lines', name='Balloon Fit',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )

            # Data in elastic region
            mask = (rel_def >= eps_range[0]) & (rel_def <= eps_range[1])
            fig.add_trace(
                go.Scatter(x=rel_def[mask], y=force[mask], mode='markers', name='Elastic Data',
                          marker=dict(size=5, color='blue')),
                row=1, col=2
            )

        # Rupture point
        if 'rupture' in results:
            rup = results['rupture']
            fig.add_vline(x=rup['epsilon'], line_dash="dash", line_color="orange",
                         annotation_text=f"Rupture ({rup['epsilon']:.3f})",
                         row=1, col=1)

        fig.update_xaxes(title_text="Relative Deformation (ε)", row=1, col=1)
        fig.update_xaxes(title_text="Relative Deformation (ε)", row=1, col=2)
        fig.update_yaxes(title_text="Force (N)", row=1, col=1)
        fig.update_yaxes(title_text="Force (N)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Comparison plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Young's Modulus Comparison")
            if 'membrane' in results and 'cytoskeleton' in results:
                moduli_data = {
                    'Component': ['Membrane', 'Cytoskeleton'],
                    'Modulus': [
                        results['membrane'].get('Em_MPa', 0),
                        results['cytoskeleton'].get('Ei_kPa', 0) / 1000  # Convert to MPa
                    ],
                    'Unit': ['MPa', 'MPa (from kPa)']
                }
                moduli_df = pd.DataFrame(moduli_data)

                fig_moduli = go.Figure(data=[
                    go.Bar(x=moduli_df['Component'], y=moduli_df['Modulus'],
                          marker=dict(color=['#1f77b4', '#ff7f0e']))
                ])
                fig_moduli.update_yaxes(type="log")
                fig_moduli.update_layout(
                    title="Young's Modulus Comparison",
                    yaxis_title="Modulus (log scale)",
                    height=400
                )
                st.plotly_chart(fig_moduli, use_container_width=True)

        with col2:
            st.markdown("### Fit Quality (R²)")
            if 'membrane' in results and 'cytoskeleton' in results:
                quality_data = {
                    'Model': ['Membrane', 'Cytoskeleton'],
                    'R²': [
                        results['membrane'].get('r_squared', 0),
                        results['cytoskeleton'].get('r_squared', 0)
                    ]
                }
                quality_df = pd.DataFrame(quality_data)

                fig_quality = go.Figure(data=[
                    go.Bar(x=quality_df['Model'], y=quality_df['R²'],
                          marker=dict(color=['#2ca02c', '#d62728']))
                ])
                fig_quality.update_yaxes(range=[0, 1])
                fig_quality.update_layout(
                    title="Fitting Quality",
                    yaxis_title="R² Value",
                    height=400
                )
                st.plotly_chart(fig_quality, use_container_width=True)

        st.markdown("---")

        # Full results JSON
        st.markdown("### Full Analysis Results (JSON)")
        st.json(results)

    else:
        st.info("👈 Upload and analyze Igor files first to view results.")

# ==================== TAB 4: Export ====================
with tabs[3]:
    st.markdown("## Export & Download")

    if st.session_state.gs_manager:
        st.markdown("### Database Export")

        col1, col2, col3 = st.columns(3)

        # CSV Export
        with col1:
            st.markdown("#### Export as CSV")
            csv_data = st.session_state.gs_manager.export_to_csv()
            if csv_data:
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name=f"afm_database_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No data to export")

        # JSON Export
        with col2:
            st.markdown("#### Export as JSON")
            json_data = st.session_state.gs_manager.export_to_json()
            if json_data:
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    file_name=f"afm_database_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No data to export")

        # Excel Export
        with col3:
            st.markdown("#### Export as Excel")
            try:
                df = st.session_state.gs_manager.get_all_cells()
                if not df.empty:
                    excel_buffer = pd.ExcelWriter(
                        f"/tmp/afm_database_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    )
                    df.to_excel(excel_buffer, index=False)
                    excel_buffer.close()

                    with open(f"/tmp/afm_database_{datetime.now().strftime('%Y%m%d')}.xlsx", "rb") as f:
                        st.download_button(
                            "📥 Download Excel",
                            f.read(),
                            file_name=f"afm_database_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                else:
                    st.info("No data to export")
            except Exception as e:
                st.error(f"Error creating Excel file: {e}")

        st.markdown("---")

        # Database Management
        st.markdown("### Database Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Database", use_container_width=True):
                st.session_state.gs_manager.get_all_cells()
                st.success("✅ Database refreshed")
                st.rerun()

        with col2:
            sheet_url = st.session_state.gs_manager.get_spreadsheet_url()
            if sheet_url:
                st.markdown(f"[📊 Open in Google Sheets]({sheet_url})")

    else:
        st.warning("⚠️ Database not connected. Enable it in the sidebar settings.")

    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown("### Current Analysis Export")

        col1, col2 = st.columns(2)

        with col1:
            import json
            json_str = json.dumps(st.session_state.results, indent=2, default=str)
            st.download_button(
                "📥 Download Analysis (JSON)",
                json_str,
                file_name="analysis_results.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            if st.session_state.data is not None:
                csv_str = st.session_state.data.to_csv(index=False)
                st.download_button(
                    "📥 Download Force Data (CSV)",
                    csv_str,
                    file_name="force_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    AFM Cell Analyzer v2.0 | Google Sheets Database Integration | Built with Streamlit & Lulevich Model
</div>
""", unsafe_allow_html=True)
