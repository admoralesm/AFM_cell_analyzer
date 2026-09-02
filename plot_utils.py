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
    # Extras that clutter a figure meant for a paper. Off means the marker or
    # caption is not drawn at all, not merely faded.
    show_video_marker: bool = True
    show_rupture_marker: bool = True
    show_schematic_moduli: bool = True
    show_legend: bool = True
    show_data: bool = True
    show_fit_line: bool = True
    show_component_heights: bool = False
    # Axis limits. None means "let plotly decide from the data". x defaults to
    # the full 0 to 1 of relative deformation so curves from different cells
    # can be laid side by side without one looking twice as squashed.
    x_range: tuple = None
    y_range: tuple = None
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
        showlegend=style.show_legend,
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

    if style.show_data:
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

    if fit_force_N is not None and style.show_fit_line:
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
        if style.show_component_heights and fy.size:
            finite = np.isfinite(fy)
            if finite.any():
                end = int(np.max(np.flatnonzero(finite)))
                fig.add_annotation(
                    x=float(fx[end]), y=float(fy[end]),
                    text=f"<b>{fy[end]:.3g} {unit_label}</b>",
                    showarrow=False, xanchor="left", yanchor="middle", xshift=6,
                    font=dict(
                        size=max(10, style.tick_size - 6), color=style.fit_color,
                        family=REGULAR_FAMILY,
                    ),
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
                if style.show_component_heights and cy.size:
                    # How high this component reaches, written where it ends.
                    # Reading a contribution off a dashed line by eye is
                    # guesswork; this states it.
                    finite = np.isfinite(cy)
                    if finite.any():
                        end = int(np.max(np.flatnonzero(finite)))
                        fig.add_annotation(
                            x=float(cx[end]), y=float(cy[end]),
                            text=f"<b>{cy[end]:.3g} {unit_label}</b>",
                            showarrow=False,
                            xanchor="left", yanchor="middle", xshift=6,
                            font=dict(
                                size=max(10, style.tick_size - 6), color=color,
                                family=REGULAR_FAMILY,
                            ),
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

    if highlight_window is not None and style.show_fit_window and not log_mode:
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

    if highlight is not None and style.show_video_marker:
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

    if rupture_epsilon is not None and style.show_rupture_marker and not log_mode:
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
    _apply_ranges(fig, style, log_mode)
    return fig


def _apply_ranges(fig, style: PlotStyle, log_mode=False):
    """
    Pin the axes when limits were asked for.

    Skipped on log axes, where a range is in decades and a linear pair of
    numbers would silently mean something else entirely.
    """
    if log_mode:
        return
    for axis, limits in (("x", style.x_range), ("y", style.y_range)):
        if not limits:
            continue
        try:
            lo, hi = float(limits[0]), float(limits[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        if axis == "x":
            fig.update_xaxes(range=[lo, hi], autorange=False)
        else:
            fig.update_yaxes(range=[lo, hi], autorange=False)


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


def _hatched_ground(fig, x0, x1, y, depth, color="#2c3e50"):
    """A fixed support drawn the way an engineering diagram draws one."""
    fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                  line=dict(color=color, width=4))
    step = (x1 - x0) / 26.0
    x = x0
    while x < x1:
        fig.add_shape(
            type="line", x0=x + step * 0.6, x1=x, y0=y, y1=y - depth,
            line=dict(color=color, width=2),
        )
        x += step


# One coil is always this tall, whatever the element. A spring drawn with a
# fixed number of coils stretches its pitch to fill whatever height it has,
# so a short element and a tall one look like different kinds of spring and
# cannot be compared by eye. Constant pitch means a shorter element simply
# has fewer coils, which is what the eye reads as "less travel".
COIL_PITCH = 7.0


def _zigzag(x, y_bottom, y_top, width, coils=None, pitch=COIL_PITCH):
    """
    Points for a spring drawn between two heights.

    ``coils`` is derived from the height so every spring in a figure shares
    the same pitch; pass a number only to override that.
    """
    span = y_top - y_bottom
    lead = span * 0.12
    body_bottom, body_top = y_bottom + lead, y_top - lead
    if coils is None:
        coils = int(round(max(2.0, (body_top - body_bottom) / max(pitch, 1e-9))))
    xs, ys = [x], [y_bottom]
    steps = max(2, int(coils) * 2)
    for i in range(steps + 1):
        ys.append(body_bottom + (body_top - body_bottom) * i / steps)
        # The parentheses matter. Without them Python reads this as
        # x + ((...) if 0 < i < steps else x), which appends 2x at the ends
        # and draws the spring leaning across the page instead of hanging
        # straight down. That was a real bug, visible as a slanted coil.
        if 0 < i < steps:
            xs.append(x + (width / 2 if i % 2 else -width / 2))
        else:
            xs.append(x)
    xs.append(x)
    ys.append(y_top)
    return xs, ys


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
    T0_mN_m=None,
    show_nucleus=True,
    show_tension=False,
    height=None,
    coupling="parallel",
    shares=None,
    break_1=None,
    break_2=None,
    membrane_mode="freeze",
    cyto_start="break",
    labels=None,
):
    """
    The model drawn as a mechanics schematic rather than a picture of a cell.

    A drawing of a squashed blob with springs in it looks like a cell but does
    not say what the model computes. This says it: a fixed support, a rigid
    platen carrying the applied force, and the elements between them in the
    arrangement the fit actually used.

    Two notations do the work that prose was doing before. An element the
    plates have not reached yet is drawn with a **gap** above its spring,
    which is how a contact that starts later is drawn anywhere in mechanics,
    and the gap is labelled with the deformation at which it closes. An
    element that has handed over and is holding what it reached is drawn with
    a **locked block** in series with its spring, meaning it takes no further
    extension.

    ``labels`` maps term name to (title, description) so a cardiomyocyte's
    parts are named as its own rather than as a generic cell's.
    """
    names = labels or {
        "tension": ("Membrane tension", ""),
        "membrane": ("Membrane", ""),
        "interior": ("Cytoskeleton", ""),
        "nucleus": ("Nucleus", ""),
    }
    epsilon = float(np.clip(epsilon, 0.0, 0.95))

    # The two membrane springs share a hue: they are one piece of material
    # answering two ways, and colouring them as unrelated parts would say the
    # opposite of what the model means.
    COLORS = {"tension": "#5dade2", "membrane": "#1f77b4",
              "interior": "#e67e22", "nucleus": "#8e44ad"}
    FADED = {"tension": "#bcdcf0", "membrane": "#a9c4d8",
             "interior": "#f2cda6", "nucleus": "#c9b0d8"}

    # What each element is doing at the deformation on display.
    if break_1 is not None and break_2 is not None:
        membrane_above = "loading" if membrane_mode == "continue" else "holding"
        cyto_below = "loading" if cyto_start == "zero" else "waiting"
        if epsilon <= break_1:
            state = {"membrane": "loading", "interior": cyto_below,
                     "nucleus": "waiting"}
        elif epsilon <= break_2:
            state = {"membrane": membrane_above, "interior": "loading",
                     "nucleus": "waiting"}
        else:
            state = {"membrane": membrane_above, "interior": "loading",
                     "nucleus": "loading"}
        # The taut network is the same shell as the elastic term, so it does
        # exactly what that one does, including going slack together with it.
        state["tension"] = state["membrane"]
        onset = {
            "tension": 0.0,
            "membrane": 0.0,
            "interior": 0.0 if cyto_start == "zero" else break_1,
            "nucleus": break_2,
        }
    else:
        state = {k: "loading" for k in COLORS}
        onset = {k: 0.0 for k in COLORS}

    terms = (
        (["tension"] if show_tension else [])
        + ["membrane", "interior"]
        + (["nucleus"] if show_nucleus else [])
    )
    moduli = {
        "tension": (T0_mN_m, "mN/m", "T<sub>0</sub>"),
        "membrane": (Em_MPa, "MPa", "E<sub>m</sub>"),
        "interior": (Ei_kPa, "kPa", "E<sub>c</sub>"),
        "nucleus": (En_kPa, "kPa", "E<sub>n</sub>"),
    }
    LAW = {"tension": "ε", "membrane": "ε³",
           "interior": "ε³ᐟ²", "nucleus": "ε³ᐟ²"}

    fig = go.Figure()
    GROUND_Y, PLATEN_Y = 20.0, 78.0
    # Labels sit below the hatching, not through it.
    LABEL_Y = GROUND_Y - 11.0
    label_size = max(10, style.tick_size - 8)
    small = max(9, style.tick_size - 10)

    _hatched_ground(fig, 4, 96, GROUND_Y, 8)
    fig.add_shape(type="rect", x0=6, x1=94, y0=PLATEN_Y, y1=PLATEN_Y + 7,
                  fillcolor="#566573", line=dict(width=0))
    fig.add_annotation(
        x=50, y=PLATEN_Y + 3.5, text="<b>cantilever</b>", showarrow=False,
        font=dict(size=small, color="white"),
    )

    # Applied force, and how far the cell has been squashed.
    fig.add_annotation(
        x=50, y=PLATEN_Y + 8, ax=50, ay=PLATEN_Y + 22,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=3,
        arrowcolor="#2c3e50", text="",
    )
    fig.add_annotation(
        x=53, y=PLATEN_Y + 17, text="<b>F</b>", showarrow=False,
        font=dict(size=label_size + 2, color="#2c3e50"),
    )
    fig.add_annotation(
        x=50, y=PLATEN_Y + 26,
        text=f"squashed to <b>ε = {epsilon:.3f}</b>"
             + (f"  ({epsilon * cell_height_um:.2f} µm of {cell_height_um:.1f})"
                if cell_height_um else ""),
        showarrow=False, font=dict(size=label_size, color="#2c3e50"),
    )

    def draw_element(term, x, y_bottom, y_top, width):
        """One spring, with a gap or a lock when the model says so."""
        colour = COLORS[term] if state[term] == "loading" else FADED[term]
        # The spring itself is always drawn solid. A dotted zigzag reads as a
        # skewed line rather than as "not engaged", and the gap symbol below
        # already carries that meaning, which is what it is for.
        top = y_top
        note = ""

        if state[term] == "waiting":
            # A contact that has not closed yet: a real gap, labelled with
            # the deformation that closes it.
            gap = (y_top - y_bottom) * 0.22
            for y in (y_top, y_top - gap):
                fig.add_shape(type="line", x0=x - width * 0.5, x1=x + width * 0.5,
                              y0=y, y1=y, line=dict(color=colour, width=3))
            note = f"gap closes at ε = {onset[term]:.2f}"
            top = y_top - gap
        elif state[term] == "holding":
            # Held at what it reached: a locked block takes no more travel.
            block = (y_top - y_bottom) * 0.13
            fig.add_shape(
                type="rect", x0=x - width * 0.45, x1=x + width * 0.45,
                y0=y_top - block, y1=y_top,
                fillcolor=colour, line=dict(color=colour, width=2),
            )
            note = "locked, no further force"
            top = y_top - block

        xs, ys = _zigzag(x, y_bottom, top, width)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                line=dict(color=colour, width=3),
            )
        )

        value, unit, symbol = moduli[term]
        caption = f"<b>{names[term][0]}</b>"
        if value is not None and style.show_schematic_moduli:
            caption += f"<br>{symbol} = {value:.3g} {unit}"
        caption += f"<br>{LAW[term]}"
        if note:
            caption += f"<br><i>{note}</i>"
        fig.add_annotation(
            x=x, y=LABEL_Y, text=caption, showarrow=False,
            yanchor="top", font=dict(size=small, color=colour),
        )

    if coupling == "series":
        # One column: same force through each, the squashes adding.
        weights = shares or {}
        total = sum(max(weights.get(t, 1.0 / len(terms)), 0.05) for t in terms)
        y = GROUND_Y
        for term in terms:
            slice_h = (PLATEN_Y - GROUND_Y) * max(
                weights.get(term, 1.0 / len(terms)), 0.05
            ) / total
            draw_element(term, 40, y, y + slice_h, 16)
            y += slice_h
        subtitle = "Stacked: same force through each, squashes add"
    else:
        spacing = 88.0 / max(len(terms), 1)
        for index, term in enumerate(terms):
            x = 6 + spacing * (index + 0.5)
            fig.add_shape(type="line", x0=x, x1=x, y0=PLATEN_Y, y1=PLATEN_Y - 3,
                          line=dict(color="#2c3e50", width=2))
            fig.add_shape(type="line", x0=x, x1=x, y0=GROUND_Y, y1=GROUND_Y + 3,
                          line=dict(color="#2c3e50", width=2))
            draw_element(term, x, GROUND_Y + 3, PLATEN_Y - 3,
                         13 if len(terms) < 4 else 10)
        subtitle = "Side by side: same squash on each, forces add"

    fig.update_xaxes(visible=False, range=[0, 100], fixedrange=True)
    fig.update_yaxes(visible=False, range=[LABEL_Y - 26, PLATEN_Y + 34],
                     fixedrange=True)
    fig.update_layout(
        height=height or max(340, int(style.height * 0.74)),
        margin=dict(l=6, r=6, t=30, b=6),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        title=dict(
            text=f"<b>{subtitle}</b>",
            font=dict(size=max(11, style.tick_size - 5), color="black"),
            x=0.5, xanchor="center",
        ),
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
