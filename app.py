"""
AFM Cell Analyzer v5 - Reorganized Workflow
Force curve analysis with plot preview before analysis
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

# ========== HELPER FUNCTION: Publication-Quality Plot ==========
def create_publication_plot(relative_def, force_N, title="Force vs Relative Deformation", force_unit="pN", line_color="#000000", line_width=4):
    """
    Create publication-quality Nature-style plot

    Parameters:
    -----------
    relative_def : array
        Relative deformation values
    force_N : array
        Force values in Newtons
    title : str
        Plot title
    force_unit : str
        Force unit: 'N', 'μN', 'nN', 'pN'
    line_color : str
        Line color (hex code)
    line_width : int
        Line width in pixels

    Returns:
    --------
    plotly figure object
    """

    # Convert force to selected unit
    unit_conversions = {
        'N': (1, 'N'),
        'μN': (1e6, 'μN'),
        'nN': (1e9, 'nN'),
        'pN': (1e12, 'pN')
    }

    conversion_factor, unit_label = unit_conversions.get(force_unit, (1e12, 'pN'))
    force_converted = force_N * conversion_factor

    # Create figure with publication-quality settings
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=relative_def,
        y=force_converted,
        mode='lines+markers',
        name='Force Curve',
        line=dict(color=line_color, width=line_width),
        marker=dict(size=8, color=line_color, line=dict(width=2, color=line_color)),
        hovertemplate='<b>ε:</b> %{x:.4f}<br><b>F:</b> %{y:.3f} ' + unit_label + '<extra></extra>'
    ))

    # Nature journal style formatting
    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color='black')),
        xaxis_title='Relative Deformation (ε)',
        yaxis_title=f'Force ({unit_label})',
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        height=600,
        margin=dict(l=100, r=50, t=100, b=100)
    )

    # Update axes with thick lines and no grid
    fig.update_xaxes(
        showline=True,
        linewidth=3,
        linecolor='black',
        showgrid=False,
        zeroline=False,
        mirror=True,
        title_font=dict(size=20, color='black'),
        tickfont=dict(size=16, color='black')
    )

    fig.update_yaxes(
        showline=True,
        linewidth=3,
        linecolor='black',
        showgrid=False,
        zeroline=False,
        mirror=True,
        title_font=dict(size=20, color='black'),
        tickfont=dict(size=16, color='black')
    )

    return fig

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
        font-size: 1.3em;
        font-weight: bold;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'gs_manager' not in st.session_state:
    st.session_state.gs_manager = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# Main header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔬 AFM Cell Analyzer</div>', unsafe_allow_html=True)
with col2:
    st.markdown("**v5.0**")

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

    # Spring constant (OPTIONAL)
    spring_constant_default = st.number_input(
        "Default Spring Constant (N/m)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.001,
        help="Optional: Spring constant (0 = not used)"
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
    "📊 Force Curve Analysis",
    "🔧 Create Force Curve (Igor)",
    "📋 Database Browser",
    "📈 Results",
    "💾 Export"
])

# ==================== TAB 1: Force Curve Analysis ====================
with tabs[0]:
    st.markdown("## Force vs Relative Deformation Curve")
    st.markdown("---")

    # ========== SECTION 1: Cell Information ==========
    st.markdown('<div class="section-header">Cell Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        cell_name = st.text_input(
            "Cell Name/ID *",
            placeholder="e.g., C2C12_001",
            help="Unique identifier for this cell"
        )

    with col2:
        date_acquired = st.date_input(
            "Date Acquired *",
            value=datetime.now().date()
        )

    with col3:
        cell_height = st.number_input(
            "Cell Height (μm) *",
            min_value=1.0,
            max_value=50.0,
            value=8.09,
            step=0.1
        )

    st.markdown("---")

    # ========== SECTION 2: Upload Force Curve ==========
    st.markdown('<div class="section-header">Upload Force vs Relative Deformation File</div>', unsafe_allow_html=True)

    st.markdown("**Expected format:** CSV or Excel with columns:")
    st.markdown("- **Relative Deformation (ε)** or similar")
    st.markdown("- **Force (nN)** or similar")

    force_curve_file = st.file_uploader(
        "Select force curve file (.csv or .xlsx)",
        type=['csv', 'xlsx'],
        key="force_curve_file"
    )

    current_data = None
    df_loaded = None
    relative_def = None
    force = None

    if force_curve_file is not None:
        try:
            # Load file
            if force_curve_file.name.endswith('.csv'):
                df_loaded = pd.read_csv(force_curve_file)
            else:
                df_loaded = pd.read_excel(force_curve_file)

            st.success(f"✅ Loaded {len(df_loaded)} data points")

            # Column selection
            col_names = df_loaded.columns.tolist()
            col1, col2 = st.columns(2)

            with col1:
                eps_col = st.selectbox(
                    "Relative Deformation Column",
                    col_names,
                    help="Select the column containing ε values"
                )

            with col2:
                force_col = st.selectbox(
                    "Force Column",
                    col_names,
                    help="Select the column containing force values"
                )

            # Extract data
            relative_def = df_loaded[eps_col].values.astype(float)
            force = df_loaded[force_col].values.astype(float)

            current_data = {
                'relative_def': relative_def,
                'force': force,
                'cell_name': cell_name,
                'date_acquired': str(date_acquired),
                'cell_height': cell_height
            }

            st.session_state.current_data = current_data

        except Exception as e:
            st.error(f"❌ File Error: {str(e)}")

    st.markdown("---")

    # ========== SECTION 3: Metadata ==========
    st.markdown('<div class="section-header">Metadata</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        video_link = st.text_input(
            "Google Drive Video Link (optional)",
            placeholder="https://drive.google.com/file/d/...",
            help="Link to compression video"
        )

    with col2:
        spring_constant = st.number_input(
            "Spring Constant (N/m) (optional)",
            min_value=0.0,
            max_value=100.0,
            value=spring_constant_default,
            step=0.001,
            help="Spring constant (0 = not used)"
        )

    st.markdown("---")

    # ========== SECTION 4: Plot Preview ==========
    if force_curve_file is not None and relative_def is not None and force is not None:
        st.markdown('<div class="section-header">Plot Preview & Customization</div>', unsafe_allow_html=True)

        # Plot customization controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            force_unit_preview = st.selectbox(
                "Force Unit",
                ["μN", "pN", "nN", "N"],
                index=0,
                label_visibility="collapsed",
                key="force_unit_preview"
            )
        with col2:
            line_color_preview = st.color_picker(
                "Line Color",
                value="#000000",
                label_visibility="collapsed",
                key="line_color_preview"
            )
        with col3:
            line_width_preview = st.slider(
                "Line Width",
                min_value=1,
                max_value=8,
                value=4,
                label_visibility="collapsed",
                key="line_width_preview"
            )
        with col4:
            st.markdown("<p style='text-align: center; color: gray; margin-top: 8px;'>Customize plot</p>", unsafe_allow_html=True)

        # Convert force to N for the plotting function (force is currently in original units)
        # Assuming input force is in nN (from CSV)
        force_N = force / 1e9  # Convert nN to N

        fig = create_publication_plot(
            relative_def,
            force_N,
            title="Force vs Relative Deformation",
            force_unit=force_unit_preview,
            line_color=line_color_preview,
            line_width=line_width_preview
        )
        st.plotly_chart(fig, use_container_width=True, key="preview_plot")

        st.markdown("---")

        # ========== SECTION 5: Analyze with Control Panel ==========
        st.markdown('<div class="section-header">Analysis</div>', unsafe_allow_html=True)

        # Create two columns: left for plot, right for controls
        col_plot, col_controls = st.columns([3, 1])

        with col_controls:
            st.markdown("### Analysis Panel")
            st.markdown("**Fitting Range (ε)**")

            # Manual range sliders
            manual_eps_min = st.slider(
                "Min ε",
                min_value=0.0,
                max_value=0.3,
                value=0.01,
                step=0.01,
                key="manual_eps_min"
            )

            manual_eps_max = st.slider(
                "Max ε",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="manual_eps_max"
            )

            st.markdown("---")

            analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

        with col_plot:
            pass  # Placeholder for now

        if analyze_button:
            if not cell_name:
                st.error("❌ Cell Name is required")
            elif cell_height is None or cell_height <= 0:
                st.error("❌ Cell Height is required (μm). Please enter a valid value and press Enter to update.")
            else:
                with st.spinner("Analyzing... (this may take a few seconds)"):
                    try:
                        # Lulevich model fitting (two-term combined model)
                        # Convert force from nN to N for the model
                        force_N_analysis = force / 1e9
                        model = LulevichModel(force_N_analysis, relative_def, cell_height)

                        # Use combined two-term fit (membrane + cytoskeleton simultaneously)
                        fit_results = model.fit_combined_elasticity(epsilon_max=manual_eps_max, epsilon_min=manual_eps_min)

                        # Check for errors and fallback to auto-detect if needed
                        if fit_results.get('success', False) == False or fit_results.get('Em_MPa', 0) == 0:
                            st.info("ℹ️ Trying auto-detected range...")
                            auto_range = model.auto_detect_elastic_range()
                            fit_results = model.fit_combined_elasticity(
                                epsilon_max=auto_range['elastic_epsilon_max'],
                                epsilon_min=auto_range['elastic_epsilon_min']
                            )

                        # Extract results
                        fit_results_membrane = {'Em_MPa': fit_results.get('Em_MPa', 0), 'r_squared': fit_results.get('r_squared', 0)}
                        fit_results_cyto = {'Ei_kPa': fit_results.get('Ei_kPa', 0), 'r_squared': fit_results.get('r_squared', 0)}

                        # Store results
                        st.session_state.results = {
                            'cell_name': cell_name,
                            'date_acquired': str(date_acquired),
                            'cell_height': cell_height,
                            'Em': fit_results_membrane['Em_MPa'],
                            'Ei': fit_results_cyto['Ei_kPa'],
                            'r2_membrane': fit_results_membrane.get('r_squared', 0),
                            'r2_cyto': fit_results_cyto.get('r_squared', 0),
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
                                'cell_height': cell_height,
                                'cantilever_constant': f"{spring_constant} N/m" if spring_constant > 0 else "N/A",
                                'Em': round(fit_results_membrane['Em_MPa'], 4),
                                'Ei': round(fit_results_cyto['Ei_kPa'], 4),
                                'video_link': video_link,
                                'force_curve_created': 'Yes',
                                'fit_quality': round(fit_results_membrane.get('r_squared', 0), 4),
                                'notes': 'Force curve analysis',
                                'analysis_status': 'Complete'
                            }
                            success, msg = gs_manager.append_cell_data(cell_data)
                            st.info(msg)

                        # Display results
                        st.markdown("---")
                        st.markdown("### Analysis Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Em (Membrane)", f"{fit_results_membrane['Em_MPa']:.2f} MPa")
                            st.metric("R² (Membrane)", f"{fit_results_membrane.get('r_squared', 0):.4f}")

                        with col2:
                            st.metric("Ei (Cytoskeleton)", f"{fit_results_cyto['Ei_kPa']:.2f} kPa")
                            st.metric("R² (Cytoskeleton)", f"{fit_results_cyto.get('r_squared', 0):.4f}")

                        # Plot with unit selector and customization
                        st.markdown("---")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.markdown("**Display Force Units:**")
                        with col2:
                            force_unit_results = st.selectbox(
                                "Force Unit",
                                ["μN", "pN", "nN", "N"],
                                index=0,
                                label_visibility="collapsed",
                                key="force_unit_results"
                            )
                        with col3:
                            line_color_results = st.color_picker(
                                "Line Color",
                                value="#000000",
                                label_visibility="collapsed",
                                key="line_color_results"
                            )
                        with col4:
                            line_width_results = st.slider(
                                "Line Width",
                                min_value=1,
                                max_value=8,
                                value=4,
                                label_visibility="collapsed",
                                key="line_width_results"
                            )
                        with col5:
                            st.markdown("<p style='text-align: center; color: gray; margin-top: 8px;'>Customize</p>", unsafe_allow_html=True)

                        # Convert force to N
                        force_N_results = force / 1e9

                        fig = create_publication_plot(
                            relative_def,
                            force_N_results,
                            title="Force vs Relative Deformation",
                            force_unit=force_unit_results,
                            line_color=line_color_results,
                            line_width=line_width_results
                        )
                        st.plotly_chart(fig, use_container_width=True, key="analysis_results_plot")

                    except Exception as e:
                        st.error(f"❌ Analysis Error: {str(e)}")

    # ========== SECTION 6: Create Force Curve from Igor ==========
    st.markdown("---")
    st.markdown("---")
    st.markdown('<div class="section-header">Create Force vs Relative Deformation from Igor Files</div>', unsafe_allow_html=True)

    st.info("Upload two Igor files (surface reference + cell compression) to generate a force curve CSV file that you can upload above.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**File 1: Surface Reference**")
        igor_surface = st.file_uploader(
            "Select surface Igor file (.ibw)",
            type=['ibw'],
            key="igor_surface",
            help="Surface/baseline measurement"
        )

    with col2:
        st.markdown("**File 2: Cell Compression**")
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
                        st.success(f"✅ {len(data_surface)} points")
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
                        st.success(f"✅ {len(data_cell)} points")
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
                cell_height_igor = st.number_input(
                    "Cell Height (μm)",
                    min_value=1.0,
                    max_value=50.0,
                    value=8.09,
                    step=0.1
                )

            st.markdown("---")

            if st.button("⚙️ Generate Force Curve from Igor", type="secondary", use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        # Baseline correction
                        baseline_corrector = BaselineCorrector(data_cell, np.arange(len(data_cell)))
                        baseline_info = baseline_corrector.auto_detect_baseline(method='flat')
                        deflection_corrected = baseline_corrector.correct_baseline()

                        # Calculate force
                        force_generated = deflection_corrected * cantilever_constant * 1e9  # Convert to pN

                        # Estimate contact point
                        contact_idx = baseline_corrector.estimate_contact_point(deflection_corrected)

                        # Calculate relative deformation
                        z_position = np.arange(len(data_cell))
                        relative_def_generated = calculate_relative_deformation(z_position, contact_idx, cell_height_igor)

                        st.success("✅ Force Curve Generated!")

                        # Create download file
                        df_output = pd.DataFrame({
                            'Relative Deformation': relative_def_generated,
                            'Force (nN)': force_generated / 1e3  # Convert pN to nN
                        })

                        csv = df_output.to_csv(index=False)

                        st.download_button(
                            label="📥 Download Force Curve (CSV)",
                            data=csv,
                            file_name="force_curve_generated.csv",
                            mime="text/csv"
                        )

                        st.markdown("---")
                        st.markdown("### Generated Data Preview")
                        st.dataframe(df_output.head(20), use_container_width=True)

                        # Plot with unit selector and customization
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.markdown("**Display Force Units:**")
                        with col2:
                            force_unit_generated = st.selectbox(
                                "Force Unit",
                                ["μN", "pN", "nN", "N"],
                                index=0,
                                label_visibility="collapsed",
                                key="force_unit_generated"
                            )
                        with col3:
                            line_color_generated = st.color_picker(
                                "Line Color",
                                value="#000000",
                                label_visibility="collapsed",
                                key="line_color_generated"
                            )
                        with col4:
                            line_width_generated = st.slider(
                                "Line Width",
                                min_value=1,
                                max_value=8,
                                value=4,
                                label_visibility="collapsed",
                                key="line_width_generated"
                            )
                        with col5:
                            st.markdown("<p style='text-align: center; color: gray; margin-top: 8px;'>Customize</p>", unsafe_allow_html=True)

                        # Convert force to N (currently in pN)
                        force_N_generated = force_generated / 1e12

                        fig = create_publication_plot(
                            relative_def_generated,
                            force_N_generated,
                            title="Generated Force vs Relative Deformation",
                            force_unit=force_unit_generated,
                            line_color=line_color_generated,
                            line_width=line_width_generated
                        )
                        st.plotly_chart(fig, use_container_width=True, key="igor_generated_plot")

                        st.info("💡 Download the CSV above and upload it in the 'Upload Force vs Relative Deformation File' section above")

                    except Exception as e:
                        st.error(f"❌ Generation Error: {str(e)}")

# ==================== TAB 2: Database Browser ====================
with tabs[1]:
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

# ==================== TAB 3: Results ====================
with tabs[2]:
    st.markdown("## Analysis Results")

    if st.session_state.results is None:
        st.info("📊 No results yet. Complete an analysis above.")
    else:
        results = st.session_state.results

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cell Information")
            st.write(f"**Cell:** {results['cell_name']}")
            st.write(f"**Date:** {results['date_acquired']}")
            st.write(f"**Height:** {results['cell_height']} μm")

        with col2:
            st.markdown("### Mechanical Properties")
            st.metric("Em (Membrane)", f"{results['Em']:.2f} MPa")
            st.metric("Ei (Cytoskeleton)", f"{results['Ei']:.2f} kPa")

        st.markdown("---")

        # Force unit selector and plot customization for results tab
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("**Display Force Units:**")
        with col2:
            force_unit_tab3 = st.selectbox(
                "Force Unit",
                ["μN", "pN", "nN", "N"],
                index=0,
                label_visibility="collapsed",
                key="force_unit_tab3"
            )
        with col3:
            line_color_tab3 = st.color_picker(
                "Line Color",
                value="#000000",
                label_visibility="collapsed",
                key="line_color_tab3"
            )
        with col4:
            line_width_tab3 = st.slider(
                "Line Width",
                min_value=1,
                max_value=8,
                value=4,
                label_visibility="collapsed",
                key="line_width_tab3"
            )
        with col5:
            st.markdown("<p style='text-align: center; color: gray; margin-top: 8px;'>Customize</p>", unsafe_allow_html=True)

        # Convert force to N
        force_N_tab3 = results['force'] / 1e9

        fig = create_publication_plot(
            results['relative_def'],
            force_N_tab3,
            title="Force vs Relative Deformation",
            force_unit=force_unit_tab3,
            line_color=line_color_tab3,
            line_width=line_width_tab3
        )
        st.plotly_chart(fig, use_container_width=True, key="tab3_results_plot")

# ==================== TAB 4: Export ====================
with tabs[3]:
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
        st.success("✅ All cells with complete metadata included")
