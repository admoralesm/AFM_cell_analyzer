"""
C2C12 Cell Compression Analysis Tool
Web interface for analyzing AFM-based single-cell compression data.
Built with Streamlit + Lulevich 2006 model.
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

from lulevich_model import LulevichModel
from video_processor import VideoProcessor
from google_drive import GoogleDriveManager

# Page config
st.set_page_config(
    page_title="C2C12 Cell Analyzer",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔬 C2C12 Cell Compression Analyzer</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**v1.0** - Based on Lulevich et al. 2006")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")

    analysis_type = st.radio(
        "Analysis Mode",
        ["Quick Analysis", "Advanced", "Batch Processing"],
        help="Quick: Data only | Advanced: Data + Video | Batch: Multiple cells"
    )

    st.markdown("---")
    st.markdown("### Cell Parameters")

    cell_height_um = st.number_input(
        "Cell Height (μm)",
        min_value=1.0,
        max_value=50.0,
        value=8.09,
        step=0.01,
        help="Initial cell height in micrometers"
    )
    cell_height_m = cell_height_um * 1e-6

    cell_radius_um = st.number_input(
        "Cell Radius (μm)",
        min_value=1.0,
        max_value=30.0,
        value=cell_height_um * 0.55,
        step=0.01,
        help="Approximate cell radius (auto-estimated if not specified)"
    )
    cell_radius_m = cell_radius_um * 1e-6

    st.markdown("---")
    st.markdown("### Fitting Parameters")

    fitting_mode = st.radio(
        "Fitting Method",
        ["Auto Detect", "Manual Range", "Advanced Optimization"],
        help="Auto: Automatic range detection | Manual: User-specified ranges"
    )

    if fitting_mode == "Manual Range":
        col1, col2 = st.columns(2)
        with col1:
            eps_min = st.number_input(
                "ε min (membrane)",
                min_value=0.0,
                max_value=0.5,
                value=0.02,
                step=0.01
            )
        with col2:
            eps_max = st.number_input(
                "ε max (membrane)",
                min_value=0.05,
                max_value=0.5,
                value=0.3,
                step=0.01
            )
    else:
        eps_min, eps_max = None, None

    st.markdown("---")
    st.markdown("### Google Drive")

    drive_enabled = st.checkbox("Enable Google Drive Storage", value=False)
    if drive_enabled:
        st.info("📌 To enable Google Drive storage: Add your Google service account credentials to Streamlit secrets.")

# Main content
tabs = st.tabs(["📊 Analysis", "📈 Visualization", "💾 Results", "ℹ️ About"])

with tabs[0]:  # Analysis tab
    st.markdown("### Upload Your Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Force-Deformation Data")
        excel_file = st.file_uploader(
            "Upload Excel file (Force vs Relative Deformation)",
            type=['xlsx', 'xls'],
            help="Excel file with columns: Force, RelDef"
        )

    with col2:
        if analysis_type != "Quick Analysis":
            st.markdown("#### Video (Optional)")
            video_file = st.file_uploader(
                "Upload compression video",
                type=['wmv', 'mp4', 'avi', 'mov'],
                help="Optional: For compression quality validation"
            )
        else:
            video_file = None

    if excel_file is not None:
        # Read Excel file
        try:
            df = pd.read_excel(excel_file)

            # Validate columns
            expected_cols = ['Force', 'RelDef']
            if not all(col in df.columns for col in expected_cols):
                st.error(f"❌ Excel must contain columns: {expected_cols}")
                st.write("Columns found:", df.columns.tolist())
            else:
                st.session_state.data = df

                # Display data preview
                st.markdown("#### Data Preview")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.dataframe(df.head(10))

                with col2:
                    st.markdown("**Data Summary**")
                    st.write(f"- Rows: {len(df)}")
                    st.write(f"- Force range: {df['Force'].min():.2e} to {df['Force'].max():.2e} N")
                    st.write(f"- Deformation range: {df['RelDef'].min():.4f} to {df['RelDef'].max():.4f}")

                st.markdown("---")

                # Video processing
                if video_file is not None and analysis_type != "Quick Analysis":
                    st.markdown("#### Video Analysis")
                    with st.spinner("Processing video..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wmv') as tmp_video:
                                tmp_video.write(video_file.read())
                                tmp_video_path = tmp_video.name

                            processor = VideoProcessor(tmp_video_path)

                            # Show progress
                            progress_bar = st.progress(0)

                            # Extract frames
                            progress_bar.progress(33)
                            frames = processor.extract_frames(step=max(1, processor.total_frames // 10))

                            # Analyze alignment
                            progress_bar.progress(66)
                            alignment = processor.analyze_compression_alignment()

                            progress_bar.progress(100)

                            # Display results
                            st.success("✅ Video analysis complete!")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Alignment Quality**")
                                st.metric(
                                    "Alignment Score",
                                    f"{alignment.get('mean_alignment_score', 0):.3f}",
                                    help="1.0 = perfectly head-on, 0.0 = completely off-axis"
                                )
                                st.metric(
                                    "Deformation Symmetry",
                                    f"{alignment.get('deformation_symmetry', 0):.3f}"
                                )

                            with col2:
                                st.markdown("**Assessment**")
                                assessment = alignment.get('quality_assessment', 'Unknown')
                                if "EXCELLENT" in assessment:
                                    st.success(f"✅ {assessment}")
                                elif "GOOD" in assessment:
                                    st.info(f"ℹ️ {assessment}")
                                elif "FAIR" in assessment:
                                    st.warning(f"⚠️ {assessment}")
                                else:
                                    st.error(f"❌ {assessment}")

                            # Cleanup
                            os.unlink(tmp_video_path)
                            processor.close()

                            # Store alignment info
                            st.session_state.video_alignment = alignment

                        except Exception as e:
                            st.error(f"❌ Video processing error: {e}")
                            st.info("Note: Video processing requires video codec support. Try uploading an MP4 file instead of WMV.")

                st.markdown("---")

                # Run analysis
                st.markdown("#### Run Lulevich Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    run_analysis = st.button("🚀 Analyze", key="run_analysis", use_container_width=True)

                with col2:
                    auto_range = st.button("📍 Auto-detect Range", key="auto_range", use_container_width=True)

                with col3:
                    st.empty()

                if auto_range:
                    st.info("Auto-detecting optimal fitting ranges...")
                    force = df['Force'].values
                    relative_def = df['RelDef'].values

                    model = LulevichModel(force, relative_def, cell_height_m, cell_radius_m)
                    auto_result = model.auto_detect_elastic_range()

                    st.success("✅ Range detected!")
                    st.json(auto_result)

                if run_analysis:
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")

                    force = df['Force'].values
                    relative_def = df['RelDef'].values

                    # Create model
                    model = LulevichModel(force, relative_def, cell_height_m, cell_radius_m)

                    # Run analyses
                    progress = st.progress(0)

                    progress.progress(25)
                    rupture = model.detect_rupture_point()

                    progress.progress(50)
                    if fitting_mode == "Auto Detect":
                        auto_range = model.auto_detect_elastic_range()
                        eps_min = auto_range['elastic_epsilon_min']
                        eps_max = auto_range['elastic_epsilon_max']
                        st.info(f"Using auto-detected range: ε ∈ [{eps_min:.4f}, {eps_max:.4f}]")
                    elif fitting_mode == "Advanced Optimization":
                        # Use rupture point to guide range
                        eps_max = min(0.3, rupture['epsilon'] * 0.8)
                        eps_min = 0.02
                        st.info(f"Using optimized range: ε ∈ [{eps_min:.4f}, {eps_max:.4f}]")

                    membrane = model.fit_membrane_elasticity(eps_min=eps_min, eps_max=eps_max)

                    progress.progress(75)
                    cytoskeleton = model.fit_cytoskeleton_elasticity(eps_min=0.05, eps_max=min(0.3, rupture['epsilon']*0.9))

                    progress.progress(100)

                    # Store results
                    st.session_state.results = model.get_summary()

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### Membrane")
                        st.metric("Young's Modulus", f"{membrane.get('Em_MPa', 'N/A'):.2f} MPa")
                        st.metric("Bending Const.", f"{membrane.get('Km_kT', 'N/A'):.1f} kT")
                        st.metric("Fit Quality (R²)", f"{membrane.get('r_squared', 'N/A'):.4f}")

                    with col2:
                        st.markdown("### Cytoskeleton")
                        st.metric("Young's Modulus", f"{cytoskeleton.get('Ei_kPa', 'N/A'):.2f} kPa")
                        st.metric("Data Points", f"{cytoskeleton.get('n_points', 'N/A')}")
                        st.metric("Fit Quality (R²)", f"{cytoskeleton.get('r_squared', 'N/A'):.4f}")

                    with col3:
                        st.markdown("### Rupture")
                        st.metric("Rupture Point (ε)", f"{rupture.get('epsilon', 'N/A'):.4f}")
                        st.metric("Rupture Force", f"{rupture.get('force', 'N/A'):.2e} N")
                        st.metric("Peaks Detected", f"{rupture.get('n_peaks_detected', 0)}")

                    st.success("✅ Analysis complete! See Visualization tab for plots.")

        except Exception as e:
            st.error(f"❌ Error reading Excel file: {e}")

    else:
        st.info("👆 Please upload an Excel file to begin analysis")

with tabs[1]:  # Visualization tab
    if st.session_state.results is not None and st.session_state.data is not None:
        df = st.session_state.data
        results = st.session_state.results

        force = df['Force'].values
        relative_def = df['RelDef'].values

        # Create model for predictions
        model = LulevichModel(force, relative_def, cell_height_m, cell_radius_m)

        # Main force-deformation plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Force vs Deformation - Full Data", "Force vs Deformation - Elastic Region")
        )

        # Full data
        fig.add_trace(
            go.Scatter(x=relative_def, y=force, mode='markers', name='Data',
                      marker=dict(size=4, color='blue')),
            row=1, col=1
        )

        # Elastic region with fit
        if 'membrane' in results:
            mem = results['membrane']
            eps_range = mem.get('epsilon_range', [0.02, 0.3])
            eps_fit = np.linspace(eps_range[0], eps_range[1], 100)
            force_fit = model.balloon_model_cubic(eps_fit, mem['Em'])

            fig.add_trace(
                go.Scatter(x=eps_fit, y=force_fit, mode='lines', name='Membrane Fit',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )

        # Data in elastic region
        mask = (relative_def >= 0.02) & (relative_def <= 0.3)
        fig.add_trace(
            go.Scatter(x=relative_def[mask], y=force[mask], mode='markers', name='Elastic Data',
                      marker=dict(size=4, color='blue')),
            row=1, col=2
        )

        # Rupture point
        if 'rupture' in results:
            rup = results['rupture']
            fig.add_vline(x=rup['epsilon'], line_dash="dash", line_color="orange",
                         annotation_text=f"Rupture ({rup['epsilon']:.4f})",
                         row=1, col=1)

        fig.update_xaxes(title_text="Relative Deformation (ε)", row=1, col=1)
        fig.update_xaxes(title_text="Relative Deformation (ε)", row=1, col=2)
        fig.update_yaxes(title_text="Force (N)", row=1, col=1)
        fig.update_yaxes(title_text="Force (N)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Additional analysis plots
        col1, col2 = st.columns(2)

        with col1:
            # Stiffness comparison
            if 'membrane' in results and 'cytoskeleton' in results:
                moduli_data = {
                    'Component': ['Membrane', 'Cytoskeleton'],
                    'Young\'s Modulus (Pa)': [
                        results['membrane'].get('Em', 0),
                        results['cytoskeleton'].get('Ei', 0) * 1000  # Convert kPa to Pa
                    ]
                }
                moduli_df = pd.DataFrame(moduli_data)

                fig_moduli = go.Figure(data=[
                    go.Bar(x=moduli_df['Component'], y=moduli_df['Young\'s Modulus (Pa)'],
                          marker=dict(color=['#1f77b4', '#ff7f0e']))
                ])
                fig_moduli.update_yaxes(type="log")
                fig_moduli.update_layout(title="Young's Modulus Comparison",
                                        yaxis_title="Young's Modulus (Pa, log scale)",
                                        height=400)
                st.plotly_chart(fig_moduli, use_container_width=True)

        with col2:
            # Fit quality
            if 'membrane' in results and 'cytoskeleton' in results:
                quality_data = {
                    'Model': ['Membrane (Balloon)', 'Cytoskeleton (Hertzian)'],
                    'R² Value': [
                        results['membrane'].get('r_squared', 0),
                        results['cytoskeleton'].get('r_squared', 0)
                    ]
                }
                quality_df = pd.DataFrame(quality_data)

                fig_quality = go.Figure(data=[
                    go.Bar(x=quality_df['Model'], y=quality_df['R² Value'],
                          marker=dict(color=['#2ca02c', '#d62728']))
                ])
                fig_quality.update_yaxes(range=[0, 1])
                fig_quality.update_layout(title="Fitting Quality (R²)",
                                         yaxis_title="R² Value",
                                         height=400)
                st.plotly_chart(fig_quality, use_container_width=True)

        # Residuals analysis
        if 'membrane' in results:
            st.markdown("### Residual Analysis (Membrane Fit)")

            mem = results['membrane']
            eps_range = mem.get('epsilon_range', [0.02, 0.3])
            mask = (relative_def >= eps_range[0]) & (relative_def <= eps_range[1])

            eps_fit = relative_def[mask]
            force_data = force[mask]
            force_pred = model.balloon_model_cubic(eps_fit, mem['Em'])
            residuals = force_data - force_pred

            col1, col2 = st.columns(2)

            with col1:
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(x=eps_fit, y=residuals, mode='markers',
                                                  marker=dict(color='purple')))
                fig_residuals.update_layout(title="Residuals vs Deformation",
                                           xaxis_title="Relative Deformation (ε)",
                                           yaxis_title="Residual (N)",
                                           height=400)
                st.plotly_chart(fig_residuals, use_container_width=True)

            with col2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=residuals, nbinsx=20,
                                               marker=dict(color='purple')))
                fig_hist.update_layout(title="Residual Distribution",
                                      xaxis_title="Residual (N)",
                                      yaxis_title="Frequency",
                                      height=400)
                st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("👈 Run analysis first to view visualizations")

with tabs[2]:  # Results tab
    if st.session_state.results is not None:
        results = st.session_state.results

        st.markdown("## Analysis Results Summary")

        # Create JSON export
        st.markdown("### Full Results (JSON)")
        st.json(results)

        # Download options
        col1, col2, col3 = st.columns(3)

        with col1:
            # JSON download
            import json
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                json_str,
                file_name="analysis_results.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            # CSV download
            results_flat = {}
            for key, val in results.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        results_flat[f"{key}_{k}"] = v
                else:
                    results_flat[key] = val

            results_csv = pd.DataFrame([results_flat]).to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                results_csv,
                file_name="analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            # Markdown summary
            if drive_enabled:
                gd = GoogleDriveManager()
                md_summary = gd.get_summary_markdown(results, "C2C12_Sample")
                st.download_button(
                    "📥 Download Summary",
                    md_summary,
                    file_name="analysis_summary.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # Google Drive upload
        if drive_enabled:
            st.markdown("---")
            st.markdown("### Upload to Google Drive")

            cell_id = st.text_input("Cell ID (for organization)", "C2C12_001")

            if st.button("☁️ Upload to Google Drive", use_container_width=True):
                gd = GoogleDriveManager()
                if gd.setup_auth():
                    filepath = gd.save_analysis_results(results, cell_id)
                    st.success(f"✅ Uploaded to Google Drive!")
                    st.write(f"Saved as: {filepath}")
                else:
                    st.error("❌ Google Drive setup failed. Check your credentials.")

    else:
        st.info("👈 Run analysis first to view results")

with tabs[3]:  # About tab
    st.markdown("""
    ## About C2C12 Cell Analyzer

    This tool provides automated analysis of AFM-based single-cell compression data
    using the Lulevich et al. (2006) model framework, tailored for C2C12 muscle cells.

    ### Key Features

    - **Lulevich Balloon Model**: Extract membrane Young's modulus from elastic region
    - **Hertzian Contact Model**: Analyze cytoskeleton elasticity
    - **Auto Range Detection**: Automatically identify optimal fitting ranges
    - **Video Quality Assessment**: Validate compression alignment and symmetry
    - **Google Drive Integration**: Store results in the cloud
    - **Publication-Ready Visualizations**: Interactive plots with Plotly

    ### References

    **Lulevich, V., Zink, T., Chen, H.-Y., Liu, F.-T., & Liu, G.-y. (2006).**
    Cell Mechanics Using Atomic Force Microscopy-Based Single-Cell Compression.
    *Langmuir*, 22(19), 8151–8155.
    https://doi.org/10.1021/la060561p

    ### Model Details

    #### Membrane Elasticity (Living Cells, ε < 0.3)

    **Equation 3 (Balloon Model):**
    $$F \\approx \\frac{2\\pi E_m h R_0 \\epsilon^3}{1 - \\nu_m}$$

    Where:
    - $E_m$ = Membrane Young's modulus
    - $h$ = Membrane thickness (4 nm)
    - $R_0$ = Cell radius
    - $\\epsilon$ = Relative deformation
    - $\\nu_m$ = Poisson ratio (0.5 for incompressible)

    **Bending Constant:**
    $$K_m = \\frac{E_m h^3}{12(1 - \\nu_m^2)}$$

    #### Cytoskeleton Elasticity (Hertzian Model)

    **Equation 6 (Hertzian Contact):**
    $$F_i = \\frac{\\sqrt{2} E_i R_0^{1/2} \\epsilon^{3/2}}{3(1 - \\nu_i^2)}$$

    Where:
    - $E_i$ = Interior (cytoskeleton) Young's modulus
    - $\\nu_i$ = Poisson ratio

    ### Tips for C2C12 Cells

    1. **Cell Height**: Measure from video or microscopy before compression
    2. **Fitting Range**: C2C12 cells may rupture earlier than lymphoma cells
      Use auto-detection or adjust manually based on your data
    3. **Video Quality**: Head-on compression gives better results for decoupling
    4. **Batch Analysis**: For multiple cells, use Batch Processing mode

    ### Data Format

    Excel file should contain two columns:
    - **Force**: In Newtons (will be normalized)
    - **RelDef**: Relative deformation (dimensionless, 0-1 range)

    Example:
    | Force (N)    | RelDef |
    |--------------|--------|
    | 1.5e-11      | 0.0001 |
    | 3.2e-11      | 0.0002 |
    | ...          | ...    |

    ### Cloud Deployment

    This app can be deployed on **Streamlit Cloud** (free tier):

    1. Push code to GitHub
    2. Sign up at https://streamlit.io/cloud
    3. Deploy directly from GitHub
    4. For Google Drive: Add credentials to Streamlit secrets

    ### Support

    For issues or feature requests, please contact the developer or submit
    feedback through the Streamlit app interface.

    ---

    **Version**: 1.0
    **Built with**: Streamlit, NumPy, SciPy, Plotly
    **License**: Open Source
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    C2C12 Cell Analyzer v1.0 | Built for UC Davis Biomedical Research
</div>
""", unsafe_allow_html=True)
