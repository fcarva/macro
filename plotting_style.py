"""Shared editorial plotting style inspired by Nexo and Flexoki."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from flexoki import Flexoki
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator


COLORS = {
    "paper": Flexoki["paper"].hex,
    "panel": Flexoki.colors.base_50.hex,
    "panel_alt": Flexoki.colors.base_100.hex,
    "grid": Flexoki.colors.base_200.hex,
    "muted": Flexoki.colors.base_300.hex,
    "axis": Flexoki.colors.base_600.hex,
    "text": Flexoki.colors.base_950.hex,
    "line_main": Flexoki.colors.cyan_400.hex,
    "line_compare": Flexoki.colors.orange_400.hex,
    "line_neutral": Flexoki.colors.blue_400.hex,
    "positive": Flexoki.colors.green_400.hex,
    "negative": Flexoki.colors.red_400.hex,
    "highlight": Flexoki.colors.orange_600.hex,
    "black": Flexoki["black"].hex,
}


SERIES_COLORS = {
    "BRA": Flexoki.colors.cyan_500.hex,
    "USA": Flexoki.colors.blue_500.hex,
    "KOR": Flexoki.colors.orange_500.hex,
    "CHN": Flexoki.colors.green_500.hex,
    "MEX": Flexoki.colors.red_400.hex,
}


def apply_plot_theme():
    """Apply a shared Flexoki-based editorial theme to matplotlib."""

    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.9,
            "grid.alpha": 0.9,
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "text.color": COLORS["text"],
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "legend.facecolor": COLORS["paper"],
            "legend.edgecolor": COLORS["paper"],
            "legend.framealpha": 1.0,
            "legend.fontsize": 9.5,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "lines.linewidth": 2.4,
            "patch.edgecolor": COLORS["paper"],
            "patch.linewidth": 0.6,
        }
    )


def ensure_path(path_like) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _swap_decimal_separator(text: str) -> str:
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_number_ptbr(value: float, decimals: int = 0, trim: bool = True) -> str:
    if np.isnan(value):
        return ""
    text = f"{value:,.{decimals}f}"
    text = _swap_decimal_separator(text)
    if trim and "," in text:
        text = text.rstrip("0").rstrip(",")
    return text


def format_percent(value: float, decimals: int = 0) -> str:
    return f"{format_number_ptbr(value, decimals=decimals)}%"


def format_pp(value: float, decimals: int = 1) -> str:
    return f"{format_number_ptbr(value, decimals=decimals)} p.p."


def format_brl_compact(value: float, decimals: int = 1) -> str:
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000_000_000:
        scaled = absolute / 1_000_000_000_000
        suffix = " tri"
    elif absolute >= 1_000_000_000:
        scaled = absolute / 1_000_000_000
        suffix = " bi"
    elif absolute >= 1_000_000:
        scaled = absolute / 1_000_000
        suffix = " mi"
    elif absolute >= 1_000:
        scaled = absolute / 1_000
        suffix = " mil"
    else:
        scaled = absolute
        suffix = ""
    return f"{sign}R$ {format_number_ptbr(scaled, decimals=decimals)}{suffix}"


def percent_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: format_percent(value, decimals=decimals))


def pp_formatter(decimals: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: format_pp(value, decimals=decimals))


def brl_compact_formatter(decimals: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: format_brl_compact(value, decimals=decimals))


def plain_number_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: format_number_ptbr(value, decimals=decimals))


def year_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: f"{int(round(value))}" if abs(value - round(value)) < 1e-6 else "")


def style_axis(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_grid: bool = False,
    y_grid: bool = True,
    zero_line: bool = False,
    integer_x: bool = False,
):
    ax.set_facecolor(COLORS["paper"])
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(length=0, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)
    if y_grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.9)
    if x_grid:
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.9)
    if zero_line:
        ax.axhline(0.0, color=COLORS["axis"], linewidth=1.0, linestyle="-", zorder=0)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def style_legend(ax, *, loc: str = "upper left", ncol: int = 1):
    legend = ax.legend(
        loc=loc,
        ncol=ncol,
        frameon=True,
        handlelength=2.8,
        labelspacing=0.9,
        handletextpad=0.8,
        borderaxespad=1.0,
        columnspacing=1.8,
        borderpad=0.75,
    )
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor(COLORS["paper"])
        frame.set_edgecolor(COLORS["grid"])
        frame.set_linewidth(0.8)
        frame.set_alpha(0.96)
        for text in legend.get_texts():
            text.set_color(COLORS["text"])
    return legend


def add_title_block(fig, title: str, subtitle: str | None = None):
    fig.suptitle(
        title,
        x=0.015,
        y=0.975,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    if subtitle:
        fig.text(
            0.015,
            0.935,
            subtitle,
            ha="left",
            va="top",
            fontsize=10.2,
            color=COLORS["axis"],
        )


def add_footer(fig, *, source: str | None = None, note: str | None = None):
    footer_lines = []
    if source:
        footer_lines.append(f"Fonte: {source}")
    if note:
        footer_lines.append(f"Obs.: {note}")
    if footer_lines:
        fig.text(
            0.015,
            0.02,
            "   ".join(footer_lines),
            ha="left",
            va="bottom",
            fontsize=8.7,
            color=COLORS["axis"],
        )


def finalize_figure(
    fig,
    output_path,
    *,
    title: str,
    subtitle: str | None = None,
    source: str | None = None,
    note: str | None = None,
    top: float = 0.88,
    bottom: float = 0.1,
):
    add_title_block(fig, title=title, subtitle=subtitle)
    add_footer(fig, source=source, note=note)
    fig.tight_layout(rect=(0, bottom, 1, top))
    return save_figure_bundle(fig, output_path)


def save_figure_bundle(fig, output_path, *, dpi: int = 220) -> Path:
    base_path = ensure_path(output_path)
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return png_path


def direct_label_last(
    ax,
    x_values,
    y_values,
    *,
    label: str,
    color: str,
    dx: float = 8,
    dy: float = 0,
    fontsize: float = 9.5,
):
    x_array = np.asarray(x_values)
    y_array = np.asarray(y_values)
    if len(x_array) == 0 or len(y_array) == 0:
        return
    ax.annotate(
        label,
        xy=(x_array[-1], y_array[-1]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=fontsize,
        color=color,
    )


def _make_flag_drawing(code: str, width: int = 18, height: int = 12) -> DrawingArea:
    drawing = DrawingArea(width, height, 0, 0)
    border = Rectangle((0, 0), width, height, facecolor="#FFFFFF", edgecolor=COLORS["grid"], linewidth=0.45)
    drawing.add_artist(border)

    if code == "BRA":
        drawing.add_artist(Rectangle((0, 0), width, height, facecolor="#009B3A", edgecolor="none"))
        drawing.add_artist(
            Polygon(
                [(width * 0.5, height * 0.92), (width * 0.88, height * 0.5), (width * 0.5, height * 0.08), (width * 0.12, height * 0.5)],
                closed=True,
                facecolor="#FFDF00",
                edgecolor="none",
            )
        )
        drawing.add_artist(Circle((width * 0.5, height * 0.5), height * 0.2, facecolor="#002776", edgecolor="none"))
    elif code == "USA":
        stripe_height = height / 7.0
        for stripe in range(7):
            if stripe % 2 == 0:
                drawing.add_artist(Rectangle((0, stripe * stripe_height), width, stripe_height, facecolor="#B22234", edgecolor="none"))
        drawing.add_artist(Rectangle((0, height * 0.45), width * 0.45, height * 0.55, facecolor="#3C3B6E", edgecolor="none"))
    elif code == "MEX":
        drawing.add_artist(Rectangle((0, 0), width / 3.0, height, facecolor="#006847", edgecolor="none"))
        drawing.add_artist(Rectangle((width / 3.0, 0), width / 3.0, height, facecolor="#FFFFFF", edgecolor="none"))
        drawing.add_artist(Rectangle((2 * width / 3.0, 0), width / 3.0, height, facecolor="#CE1126", edgecolor="none"))
        drawing.add_artist(Circle((width * 0.5, height * 0.5), height * 0.08, facecolor="#9C7C38", edgecolor="none"))
    elif code == "CHN":
        drawing.add_artist(Rectangle((0, 0), width, height, facecolor="#DE2910", edgecolor="none"))
        drawing.add_artist(Circle((width * 0.27, height * 0.72), height * 0.12, facecolor="#FFDE00", edgecolor="none"))
    elif code == "KOR":
        drawing.add_artist(Rectangle((0, 0), width, height, facecolor="#FFFFFF", edgecolor="none"))
        drawing.add_artist(Circle((width * 0.5, height * 0.58), height * 0.18, facecolor="#CD2E3A", edgecolor="none"))
        drawing.add_artist(Circle((width * 0.5, height * 0.42), height * 0.18, facecolor="#0047A0", edgecolor="none"))
    else:
        drawing.add_artist(Rectangle((0, 0), width, height, facecolor=COLORS["panel_alt"], edgecolor="none"))

    return drawing


def make_country_badge(code: str, color: str, *, fontsize: float = 9.2):
    return HPacker(
        children=[
            _make_flag_drawing(code),
            TextArea(
                code,
                textprops={
                    "color": color,
                    "fontsize": fontsize,
                    "fontweight": "bold",
                    "va": "center",
                },
            ),
        ],
        align="center",
        pad=0,
        sep=3,
    )


def add_country_badge(
    ax,
    *,
    x: float,
    y: float,
    code: str,
    color: str,
    dx: float = 8,
    dy: float = 0,
    with_connector: bool = False,
    connector_color: str | None = None,
):
    badge = make_country_badge(code, color)
    annotation = AnnotationBbox(
        badge,
        (x, y),
        xybox=(dx, dy),
        xycoords="data",
        boxcoords="offset points",
        frameon=False,
        box_alignment=(0.0, 0.5),
        pad=0.0,
        arrowprops={"arrowstyle": "-", "color": connector_color or color, "lw": 1.0, "alpha": 0.7} if with_connector else None,
    )
    ax.add_artist(annotation)
    return annotation


def place_country_end_labels(
    ax,
    series: list[dict],
    *,
    min_gap_frac: float = 0.085,
    x_pad_frac: float = 0.1,
):
    if not series:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-9)
    x_range = max(x_max - x_min, 1e-9)
    min_gap = y_range * min_gap_frac
    label_x = x_max + x_range * x_pad_frac
    lower_bound = y_min + min_gap * 1.15
    upper_bound = y_max - min_gap * 1.15

    ordered = sorted(series, key=lambda item: item["y"])
    if len(ordered) > 1:
        available_span = max(upper_bound - lower_bound, min_gap)
        max_gap = available_span / (len(ordered) - 1)
        min_gap = min(min_gap, max_gap)

    adjusted = [float(np.clip(item["y"], lower_bound, upper_bound)) for item in ordered]

    for index in range(1, len(adjusted)):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + min_gap)

    if adjusted:
        overflow = adjusted[-1] - upper_bound
        if overflow > 0:
            adjusted = [value - overflow for value in adjusted]

    for index in range(len(adjusted) - 2, -1, -1):
        adjusted[index] = min(adjusted[index], adjusted[index + 1] - min_gap)

    if adjusted:
        underflow = lower_bound - adjusted[0]
        if underflow > 0:
            adjusted = [value + underflow for value in adjusted]

    ax.set_xlim(x_min, x_max + x_range * (x_pad_frac + 0.2))

    for item, y_target in zip(ordered, adjusted):
        connector_end = label_x - x_range * 0.03
        ax.plot(
            [item["x"], connector_end],
            [item["y"], y_target],
            color=item["color"],
            linewidth=1.05,
            alpha=0.5,
            zorder=3,
        )
        ax.scatter([item["x"]], [item["y"]], color=item["color"], s=14, zorder=4)
        add_country_badge(
            ax,
            x=label_x,
            y=y_target,
            code=item["code"],
            color=item["color"],
            dx=0,
            dy=0,
        )


apply_plot_theme()
