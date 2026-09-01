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
    marker_size: int = 7
    line_width: int = 4
    height: int = 560
    show_grid: bool = False
    axis_width: int = 4
    axis_title_size: int = 28
    tick_size: int = 22
    title_size: int = 24
    bold_axes: bool = True
    log_scale: bool = False
    show_fit_window: bool = True
    show_components: bool = True
    template: str = "publication"

    # Legacy alias: older call sites used a single `font_size`.
    @property
    def font_size(self):
        return self.tick_size

    def as_dict(self):
        return asdict(self)


# Plotly's `weight` font attribute is recent; a black font family gives bold
# tick labels on every version.
BOLD_FAMILY = "Arial Black, Arial Bold, Helvetica, sans-serif"
REGULAR_FAMILY = "Arial, Helvetica, sans-serif"


def _bold(text, enabled=True):
    """Plotly renders a small HTML subset in titles."""
    return f"<b>{text}</b>" if enabled else text


def _style_axes(fig, style: PlotStyle, x_title, y_title, log_x=False, log_y=False):
    family = BOLD_FAMILY if style.bold_axes else REGULAR_FAMILY
    common = dict(
        showline=True,
        linewidth=style.axis_width,
        linecolor="black",
        mirror=True,
        showgrid=style.show_grid,
        gridcolor="#e6e6e6",
        gridwidth=1,
        zeroline=False,
        ticks="outside",
        tickwidth=style.axis_width,
        ticklen=max(8, style.axis_width * 3),
        tickcolor="black",
        title_font=dict(size=style.axis_title_size, color="black", family=family),
        tickfont=dict(size=style.tick_size, color="black", family=family),
        automargin=True,
    )
    fig.update_xaxes(
        title_text=_bold(x_title, style.bold_axes),
        type="log" if log_x else "linear",
        **common,
    )
    fig.update_yaxes(
        title_text=_bold(y_title, style.bold_axes),
        type="log" if log_y else "linear",
        **common,
    )


def _base_layout(fig, style: PlotStyle, title):
    # Bigger fonts need proportionally bigger margins or the axis titles clip.
    side = 60 + int(2.6 * style.axis_title_size)
    bottom = 50 + int(2.4 * style.axis_title_size)
    fig.update_layout(
        title=dict(
            text=_bold(title, style.bold_axes),
            font=dict(size=style.title_size, color="black", family=REGULAR_FAMILY),
            x=0.02,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        height=style.height,
        margin=dict(l=side, r=40, t=max(70, style.title_size * 3), b=bottom),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=max(11, style.tick_size - 6), family=REGULAR_FAMILY),
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
    nucleus_N=None,
    fit_window=None,
    rupture_epsilon=None,
    highlight=None,
    highlight_window=None,
):
    """
    One figure for both preview and results.

    Passing ``fit_force_N`` adds the fitted curve. The component curves are
    labelled as contributions to that one total rather than as separate fits,
    because that is what they are: a single fit whose terms are drawn apart to
    show which structure is carrying the force where.
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
                name="Model",
                line=dict(color=style.fit_color, width=style.line_width, dash="dash"),
                hovertemplate="ε = %{x:.4f}<br>F(model) = %{y:.4g} " + unit_label + "<extra></extra>",
            )
        )

        if style.show_components:
            for comp, label, dash, color in (
                (membrane_N, "· membrane contribution", "dash", "#2ca02c"),
                (interior_N, "· cytoskeleton contribution", "dot", "#9467bd"),
                (nucleus_N, "· nucleus contribution", "dashdot", "#e377c2"),
            ):
                if comp is None or not np.any(np.asarray(comp)):
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
        # Accept a single (lo, hi) or a list of windows, each optionally
        # carrying its own label and colour, so the sequential fit can show
        # the cytoskeleton and membrane windows separately.
        windows = fit_window
        if len(windows) == 2 and np.isscalar(windows[0]):
            windows = [{"range": tuple(fit_window), "label": "fit window"}]
        for i, win in enumerate(windows):
            if not isinstance(win, dict):
                win = {"range": tuple(win)}
            lo, hi = win["range"]
            fig.add_vrect(
                x0=lo,
                x1=hi,
                fillcolor=win.get("color", ("#2ca02c", "#9467bd")[i % 2]),
                opacity=win.get("opacity", 0.12),
                layer="below",
                line_width=0,
                annotation_text=win.get("label", "fit window"),
                annotation_position="bottom left",
                annotation_font_size=max(10, style.tick_size - 6),
            )

    if highlight_window is not None and not log_mode:
        # The segment currently being edited in the range table. It is drawn
        # over the fit windows rather than under them so that the stretch the
        # user is typing numbers for is unmistakable on the curve.
        try:
            hw_lo, hw_hi = float(highlight_window[0]), float(highlight_window[1])
            hw_label = (
                highlight_window[2] if len(highlight_window) > 2 else "selected range"
            )
        except (TypeError, ValueError, IndexError):
            hw_lo = hw_hi = None
        if hw_lo is not None and hw_hi > hw_lo:
            fig.add_vrect(
                x0=hw_lo,
                x1=hw_hi,
                fillcolor="#ffd166",
                opacity=0.34,
                layer="below",
                line=dict(color="#e8a600", width=3),
                annotation_text=str(hw_label),
                annotation_position="top left",
                annotation_font_size=max(11, style.tick_size - 4),
                annotation_font_color="#8a6100",
            )

    if highlight is not None:
        # The point on the curve that the displayed video frame corresponds to.
        hx, hy_N = highlight
        hy, _ = from_newtons(hy_N, style.force_unit)
        fig.add_trace(
            go.Scatter(
                x=[hx],
                y=[float(hy)],
                mode="markers",
                name="video frame",
                marker=dict(
                    size=style.marker_size + 12,
                    color="rgba(255,165,0,0.9)",
                    symbol="circle-open",
                    line=dict(width=4, color="#ff7f0e"),
                ),
                hovertemplate="ε = %{x:.4f}<br>F = %{y:.4g}<extra>video frame</extra>",
            )
        )
        if not log_mode:
            fig.add_vline(x=hx, line=dict(color="#ff7f0e", width=2, dash="dot"))

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


# ------------------------------------------------------------- schematic


def _spring_path(x_center, y_bottom, y_top, width, n_coils=6):
    """Zigzag points for a drawn spring between two heights."""
    n = max(2, int(n_coils)) * 2
    ys = np.linspace(y_bottom, y_top, n + 3)
    xs = [x_center]
    for i in range(1, n + 2):
        xs.append(x_center + (width / 2 if i % 2 else -width / 2))
    xs.append(x_center)
    return list(xs[: len(ys)]), list(ys)


def _add_spring(fig, x_center, y_bottom, y_top, width, color, line_width, n_coils=6):
    xs, ys = _spring_path(x_center, y_bottom, y_top, width, n_coils)
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=line_width),
            hoverinfo="skip", showlegend=False,
        )
    )


def cell_schematic(
    style: PlotStyle,
    epsilon=0.0,
    cell_height_um=8.0,
    cell_radius_um=4.4,
    nucleus_radius_um=1.5,
    membrane_thickness_nm=4.0,
    nucleus_onset=None,
    Em_MPa=None,
    Ei_kPa=None,
    En_kPa=None,
    show_nucleus=True,
    height=None,
    coupling="parallel",
    shares=None,
    break_1=None,
    break_2=None,
    membrane_mode="freeze",
    cyto_start="break",
):
    """
    A small side-on diagram of what is being modelled.

    Draws the cantilever, the cell squashed to the current deformation, and
    the parts the three terms describe. The cell is drawn to the real aspect
    ratio implied by the geometry settings, so a wrong cell height or radius
    is visible as a shape that does not look like the cell in the video.
    Volume is held roughly constant as it flattens, which is why it widens.

    ``coupling`` changes the arrangement, because that is the whole point of
    the setting: in parallel the cytoskeleton spring sits inside the membrane
    balloon and both are squashed together, while in series the elements are
    stacked between the plates and each takes its own slice of the squash.
    ``shares`` maps element name to its fraction of the total deformation and
    is used to size the stacked elements.

    With ``break_1`` and ``break_2`` given, the diagram also shows which
    elements are carrying load at the deformation on display: the one that is
    taking up the squash right now is drawn solid, one that has handed over
    and is merely holding what it reached is greyed, and one that has not been
    reached yet is faint and dashed. That is the difference between the first
    two thirds of a compression and what happens once the nucleus is met.
    """
    epsilon = float(np.clip(epsilon, 0.0, 0.95))
    h = cell_height_um * (1.0 - epsilon)
    # Constant volume for an oblate shape: R grows as 1/sqrt(1 - eps).
    r = cell_radius_um / np.sqrt(max(1.0 - epsilon, 1e-3))
    nucleus_engaged = nucleus_onset is not None and epsilon >= nucleus_onset

    # Keep the drawing filling the panel: the axes are aspect-locked, so an
    # over-wide x range shrinks the cell into the middle of a lot of white.
    span = max(cell_radius_um * 1.7, r * 1.35)
    fig = go.Figure()

    # Substrate.
    fig.add_shape(
        type="rect", x0=-span, x1=span, y0=-cell_height_um * 0.16, y1=0,
        fillcolor="#7f8c8d", line=dict(width=0), layer="below",
    )
    # Cantilever, resting on top of the cell.
    fig.add_shape(
        type="rect", x0=-span * 0.75, x1=span * 0.75, y0=h, y1=h + cell_height_um * 0.13,
        fillcolor="#566573", line=dict(width=0),
    )

    membrane_line = max(3, membrane_thickness_nm * 0.9)

    # Which elements are doing what at this deformation. The two composition
    # settings decide it: whether the membrane keeps stiffening past the first
    # boundary, and whether the cytoskeleton was already loaded before it.
    if break_1 is not None and break_2 is not None:
        membrane_above = "loading" if membrane_mode == "continue" else "holding"
        cyto_below = "loading" if cyto_start == "zero" else "waiting"
        if epsilon <= break_1:
            state = {"membrane": "loading", "interior": cyto_below, "nucleus": "waiting"}
            stage_name = (
                "Membrane and cytoskeleton" if cyto_below == "loading"
                else "Membrane alone"
            )
        elif epsilon <= break_2:
            state = {
                "membrane": membrane_above, "interior": "loading", "nucleus": "waiting",
            }
            stage_name = (
                "Membrane and cytoskeleton" if membrane_above == "loading"
                else "Cytoskeleton, membrane holding"
            )
        else:
            state = {
                "membrane": membrane_above, "interior": "loading", "nucleus": "loading",
            }
            stage_name = (
                "All three together" if membrane_above == "loading"
                else "Cytoskeleton and nucleus together"
            )
    else:
        state = {"membrane": "loading", "interior": "loading", "nucleus": "loading"}
        stage_name = None

    LOADING = {"membrane": "#1f77b4", "interior": "#e67e22", "nucleus": "#8e44ad"}
    FADED = {"membrane": "#aebfcc", "interior": "#f0c9a0", "nucleus": "#cbb2d6"}

    def colour(name):
        return LOADING[name] if state[name] == "loading" else FADED[name]

    def dash(name):
        return "dot" if state[name] == "waiting" else None

    if coupling == "series":
        # Stacked between the plates: each element takes its own slice of the
        # total squash, sized by its share of the deformation.
        weights = shares or {}
        order = [("membrane", "#1f77b4")]
        if show_nucleus:
            order.append(("nucleus", "#8e44ad"))
        order.append(("interior", "#e67e22"))
        total = sum(max(weights.get(k, 1.0 / len(order)), 0.02) for k, _ in order)
        y = 0.0
        for name, _ in order:
            slice_h = h * max(weights.get(name, 1.0 / len(order)), 0.02) / total
            if name == "membrane":
                fig.add_shape(
                    type="rect", x0=-r * 0.9, x1=r * 0.9, y0=y, y1=y + slice_h,
                    fillcolor="rgba(214,234,248,0.95)",
                    line=dict(color=colour("membrane"), width=membrane_line),
                )
            elif name == "interior":
                _add_spring(fig, 0.0, y, y + slice_h, r * 0.85, colour("interior"), 4, n_coils=5)
            else:
                nr = min(nucleus_radius_um, r * 0.75)
                fig.add_shape(
                    type="circle", x0=-nr, x1=nr, y0=y, y1=y + slice_h,
                    fillcolor="rgba(155,89,182,0.75)"
                    if state["nucleus"] == "loading" else "rgba(155,89,182,0.25)",
                    line=dict(color=colour("nucleus"), width=2),
                )
            y += slice_h
    else:
        # The membrane balloon, with one cytoskeleton spring inside it and the
        # nucleus drawn as a spring of its own.
        fig.add_shape(
            type="circle", x0=-r, x1=r, y0=0, y1=h,
            fillcolor="rgba(214,234,248,0.95)",
            line=dict(color=colour("membrane"), width=membrane_line,
                      dash=dash("membrane")),
            layer="below",
        )
        _add_spring(fig, -r * 0.50, h * 0.15, h * 0.85, r * 0.30,
                    colour("interior"), 3, n_coils=5)

        if show_nucleus:
            nr = min(nucleus_radius_um / np.sqrt(max(1.0 - epsilon, 1e-3)), r * 0.40)
            nh = min(nucleus_radius_um * 2.0 * (1.0 - epsilon * 0.6), h * 0.72)
            engaged = state["nucleus"] == "loading"
            fig.add_shape(
                type="circle",
                x0=r * 0.16, x1=r * 0.16 + 2 * nr,
                y0=h / 2 - nh / 2, y1=h / 2 + nh / 2,
                fillcolor="rgba(155,89,182,0.30)" if engaged else "rgba(155,89,182,0.10)",
                line=dict(color=colour("nucleus"), width=2, dash="dot"),
                layer="below",
            )
            _add_spring(
                fig, r * 0.16 + nr, h * 0.17, h * 0.83, min(nr * 1.2, r * 0.34),
                colour("nucleus"), 3, n_coils=5,
            )

    # Height marker.
    fig.add_annotation(
        x=r * 1.28, y=h / 2, ax=r * 1.28, ay=h,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#2c3e50",
        text="",
    )
    fig.add_annotation(
        x=r * 1.28, y=h / 2, ax=r * 1.28, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#2c3e50",
        text="",
    )
    fig.add_annotation(
        x=r * 1.34, y=h / 2, text=f"<b>{h:.1f} µm</b>", showarrow=False,
        xanchor="left", font=dict(size=max(10, style.tick_size - 6), color="#2c3e50"),
    )

    WORD = {"loading": "carrying load", "holding": "holding, no new force",
            "waiting": "not reached yet"}
    labels = []
    if Em_MPa is not None:
        labels.append(
            f"<b>Membrane</b>  E<sub>m</sub> = {Em_MPa:.3g} MPa "
            f"<i>({WORD[state['membrane']]})</i>"
        )
    if Ei_kPa is not None:
        labels.append(
            f"<b>Cytoskeleton</b>  E<sub>c</sub> = {Ei_kPa:.3g} kPa "
            f"<i>({WORD[state['interior']]})</i>"
        )
    if show_nucleus and En_kPa is not None:
        labels.append(
            f"<b>Nucleus</b>  E<sub>n</sub> = {En_kPa:.3g} kPa "
            f"<i>({WORD[state['nucleus']]})</i>"
        )
    caption = f"ε = {epsilon:.3f}"
    if labels:
        caption += "<br>" + "<br>".join(labels)

    # Centred on the axes, not on x = 0: the drawing is offset to leave room
    # for the height marker, so a caption centred at the origin runs off the
    # left edge and loses its first characters.
    fig.add_annotation(
        x=span * 0.225, y=-cell_height_um * 0.24, text=caption, showarrow=False,
        xanchor="center", yanchor="top", align="center",
        font=dict(size=max(10, style.tick_size - 7), color="#2c3e50"),
    )

    fig.update_xaxes(
        visible=False, range=[-span, span * 1.45],
        scaleanchor="y", scaleratio=1, fixedrange=True,
    )
    fig.update_yaxes(
        visible=False,
        range=[-cell_height_um * 0.80, cell_height_um * 1.10],
        fixedrange=True,
    )
    fig.update_layout(
        height=height or max(300, int(style.height * 0.66)),
        margin=dict(l=6, r=6, t=28, b=6),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        title=dict(text="<b>%s</b>" % (
            stage_name if stage_name
            else "Stacked in series" if coupling == "series"
            else "Spring inside the balloon" if coupling == "parallel"
            else "Hybrid load path"),
                   font=dict(size=max(12, style.tick_size - 4), color="black"), x=0.5,
                   xanchor="center"),
    )
    return fig


def exponent_profile_figure(profile, style: PlotStyle, break_1=None, break_2=None,
                            title="What power law is the curve following?"):
    """
    Local log-log slope against deformation.

    The two reference lines are the exponents the model assumes: 3 where the
    membrane carries the load, 3/2 for a Hertzian contact. Where the measured
    curve sits on one of them, that stage of the model is the right shape for
    the data; where it wanders, it is not.
    """
    eps = np.asarray(profile.get("epsilon", []), dtype=float)
    exponent = np.asarray(profile.get("exponent", []), dtype=float)
    fig = go.Figure()
    if eps.size:
        fig.add_trace(
            go.Scatter(
                x=eps, y=exponent, mode="lines+markers",
                name="measured exponent",
                line=dict(color=style.data_color, width=style.line_width),
                marker=dict(size=max(3, style.marker_size - 2)),
                hovertemplate="ε = %{x:.3f}<br>exponent = %{y:.2f}<extra></extra>",
            )
        )
    for value, label, colour in ((3.0, "ε³ membrane", "#2ca02c"),
                                 (1.5, "ε³ᐟ² Hertzian", "#9467bd")):
        # Inside the axes, not to the right of them: an annotation hung off the
        # right edge is the first thing to be clipped.
        fig.add_hline(
            y=value, line=dict(color=colour, width=2, dash="dash"),
            annotation_text=label, annotation_position="top left",
            annotation_font_size=max(10, style.tick_size - 8),
            annotation_font_color=colour,
        )
    for value, label, colour in ((break_1, "ε₁", "#2ca02c"), (break_2, "ε₂", "#e377c2")):
        if value is not None:
            fig.add_vline(
                x=float(value), line=dict(color=colour, width=2, dash="dot"),
                annotation_text=label, annotation_position="top",
                annotation_font_size=max(10, style.tick_size - 6),
            )
    _base_layout(fig, style, title)
    fig.update_layout(
        height=max(360, int(style.height * 0.78)),
        showlegend=False,
        margin=dict(r=70),
    )
    _style_axes(fig, style, "Relative deformation, ε", "Local exponent")
    fig.update_yaxes(range=[0, 4.4], dtick=1)
    return fig
