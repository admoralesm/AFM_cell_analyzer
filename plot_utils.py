"""
Plot construction for the AFM Cell Analyzer.

Every figure in the app is built here from one PlotStyle object, so the
display settings live in a single place instead of being re-declared next to
each chart. Force is always passed in NEWTONS and converted for display at
the last moment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import plotly.graph_objects as go

# Display unit -> (factor applied to newtons, axis label)
FORCE_UNITS = {
    "N": (1.0, "N"),
    "mN": (1e3, "mN"),
    "μN": (1e6, "μN"),
    "nN": (1e9, "nN"),
    "pN": (1e12, "pN"),
}

# Input unit -> factor to newtons
INPUT_FORCE_UNITS = {
    "N (newtons)": 1.0,
    "mN (millinewtons)": 1e-3,
    "μN (micronewtons)": 1e-6,
    "nN (nanonewtons)": 1e-9,
    "pN (piconewtons)": 1e-12,
}


def to_newtons(values, input_unit):
    return np.asarray(values, dtype=float) * INPUT_FORCE_UNITS.get(input_unit, 1e-9)


def from_newtons(values_N, display_unit):
    factor, label = FORCE_UNITS.get(display_unit, (1e9, "nN"))
    return np.asarray(values_N, dtype=float) * factor, label


def autoscale_unit(values_N):
    """Pick the display unit that puts the peak force in a readable 1-1000 range."""
    peak = float(np.nanmax(np.abs(values_N))) if np.size(values_N) else 0.0
    if peak <= 0:
        return "nN"
    for unit in ("N", "mN", "μN", "nN", "pN"):
        factor, _ = FORCE_UNITS[unit]
        if 1.0 <= peak * factor < 1000.0:
            return unit
    return "nN"


@dataclass
class PlotStyle:
    """Everything the user can change about how a figure looks."""

    force_unit: str = "nN"
    data_color: str = "#1f77b4"
    fit_color: str = "#d62728"
    marker_size: int = 6
    line_width: int = 3
    height: int = 520
    show_grid: bool = False
    axis_width: int = 2
    font_size: int = 16
    title_size: int = 20
    log_scale: bool = False
    show_fit_window: bool = True
    show_components: bool = True
    template: str = "publication"

    def as_dict(self):
        return asdict(self)


def _style_axes(fig, style: PlotStyle, x_title, y_title, log_x=False, log_y=False):
    common = dict(
        showline=True,
        linewidth=style.axis_width,
        linecolor="black",
        mirror=True,
        showgrid=style.show_grid,
        gridcolor="#e6e6e6",
        zeroline=False,
        ticks="outside",
        tickwidth=style.axis_width,
        ticklen=6,
        title_font=dict(size=style.font_size + 3, color="black"),
        tickfont=dict(size=style.font_size, color="black"),
    )
    fig.update_xaxes(title_text=x_title, type="log" if log_x else "linear", **common)
    fig.update_yaxes(title_text=y_title, type="log" if log_y else "linear", **common)


def _base_layout(fig, style: PlotStyle, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=style.title_size, color="black"), x=0.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        height=style.height,
        margin=dict(l=90, r=40, t=70, b=80),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=style.font_size - 2),
        ),
    )


def force_curve_figure(
    epsilon,
    force_N,
    style: PlotStyle,
    title="Force vs relative deformation",
    fit_epsilon=None,
    fit_force_N=None,
    membrane_N=None,
    interior_N=None,
    fit_window=None,
    rupture_epsilon=None,
):
    """
    One figure for both preview and results.

    Passing ``fit_force_N`` adds the fitted curve; passing ``membrane_N`` /
    ``interior_N`` adds the two model components as dashed lines so it is
    visible which term is carrying the force.
    """
    epsilon = np.asarray(epsilon, dtype=float)
    y, unit_label = from_newtons(force_N, style.force_unit)

    log_mode = style.log_scale
    fig = go.Figure()

    if log_mode:
        # Log axes cannot show non-positive values; drop them rather than
        # letting plotly silently blank the trace.
        keep = (epsilon > 0) & (y > 0)
        epsilon_plot, y_plot = epsilon[keep], y[keep]
    else:
        epsilon_plot, y_plot = epsilon, y

    fig.add_trace(
        go.Scatter(
            x=epsilon_plot,
            y=y_plot,
            mode="markers",
            name="Experimental data",
            marker=dict(
                size=style.marker_size,
                color=style.data_color,
                line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            ),
            hovertemplate="ε = %{x:.4f}<br>F = %{y:.4g} " + unit_label + "<extra></extra>",
        )
    )

    if fit_force_N is not None:
        fx = np.asarray(fit_epsilon if fit_epsilon is not None else epsilon, dtype=float)
        fy, _ = from_newtons(fit_force_N, style.force_unit)
        if log_mode:
            keep = (fx > 0) & (fy > 0)
            fx, fy = fx[keep], fy[keep]
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=fy,
                mode="lines",
                name="Lulevich fit",
                line=dict(color=style.fit_color, width=style.line_width),
                hovertemplate="ε = %{x:.4f}<br>F(fit) = %{y:.4g} " + unit_label + "<extra></extra>",
            )
        )

        if style.show_components:
            for comp, label, dash, color in (
                (membrane_N, "Membrane term (ε³)", "dash", "#2ca02c"),
                (interior_N, "Interior term (ε³ᐟ²)", "dot", "#9467bd"),
            ):
                if comp is None:
                    continue
                cy, _ = from_newtons(comp, style.force_unit)
                cx = np.asarray(fit_epsilon if fit_epsilon is not None else epsilon, dtype=float)
                if log_mode:
                    keep = (cx > 0) & (cy > 0)
                    cx, cy = cx[keep], cy[keep]
                fig.add_trace(
                    go.Scatter(
                        x=cx,
                        y=cy,
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=max(1, style.line_width - 1), dash=dash),
                        hovertemplate="ε = %{x:.4f}<br>%{y:.4g} " + unit_label + "<extra></extra>",
                    )
                )

    if fit_window is not None and style.show_fit_window and not log_mode:
        lo, hi = fit_window
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor="#2ca02c",
            opacity=0.10,
            layer="below",
            line_width=0,
            annotation_text="fit window",
            annotation_position="top left",
            annotation_font_size=style.font_size - 3,
        )

    if rupture_epsilon is not None and not log_mode:
        fig.add_vline(
            x=rupture_epsilon,
            line=dict(color="#ff7f0e", width=2, dash="dashdot"),
            annotation_text="rupture",
            annotation_position="top right",
            annotation_font_size=style.font_size - 3,
        )

    _base_layout(fig, style, title)
    _style_axes(
        fig,
        style,
        "Relative deformation, ε",
        f"Force ({unit_label})",
        log_x=log_mode,
        log_y=log_mode,
    )
    return fig


def residual_figure(epsilon, residual_N, style: PlotStyle, title="Fit residuals"):
    y, unit_label = from_newtons(residual_N, style.force_unit)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epsilon,
            y=y,
            mode="markers",
            name="Residual",
            marker=dict(size=style.marker_size, color=style.data_color),
            hovertemplate="ε = %{x:.4f}<br>ΔF = %{y:.4g} " + unit_label + "<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="black", width=1))
    _base_layout(fig, style, title)
    fig.update_layout(height=max(220, style.height // 2), showlegend=False)
    _style_axes(fig, style, "Relative deformation, ε", f"Data − fit ({unit_label})")
    return fig


def sensitivity_figure(trials, style: PlotStyle, title="Sensitivity to fit window"):
    """Em and Ei as a function of the upper bound of the fit window."""
    if not trials:
        return None
    x = [t["epsilon_max"] for t in trials]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[t["Em_MPa"] for t in trials],
            mode="lines+markers",
            name="Em (MPa)",
            line=dict(color="#2ca02c", width=style.line_width),
            marker=dict(size=style.marker_size + 1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[t["Ei_kPa"] for t in trials],
            mode="lines+markers",
            name="Ei (kPa)",
            yaxis="y2",
            line=dict(color="#9467bd", width=style.line_width, dash="dot"),
            marker=dict(size=style.marker_size + 1),
        )
    )
    _base_layout(fig, style, title)
    fig.update_layout(
        height=max(260, style.height // 2),
        yaxis2=dict(
            title="Ei (kPa)",
            overlaying="y",
            side="right",
            showline=True,
            linewidth=style.axis_width,
            linecolor="black",
            title_font=dict(size=style.font_size + 1),
            tickfont=dict(size=style.font_size - 1),
        ),
    )
    _style_axes(fig, style, "Upper bound of fit window, ε_max", "Em (MPa)")
    fig.update_yaxes(mirror=False)
    return fig
