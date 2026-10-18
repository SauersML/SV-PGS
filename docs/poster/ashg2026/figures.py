"""The science poster's figures and its model equation, as print-resolution PNGs for the PowerPoint template.

Every value is the repository's recorded measurement:
- the simulation study: simulation_summary.json, collected by simulation.py from the power grid's finished tasks
  (benchmarks/power/grid.py on bench-sim's 1000 Genomes chr22 cohort, MSI). Until a figure's grid points have results,
  its dashed placeholder is drawn instead.

    python3 figures.py [simulation_summary.json] [output directory]

Each figure is drawn in an SVG whose units are hundredths of an inch at the figure's printed size (the template's picture
frame), so 1 pt of type is 100 / 72 units. Chrome renders it at 2.5x (240 dpi).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PT = 100.0 / 72.0

NAVY = "#235078"
TEAL = "#1482A5"
INK = "#1d1d1f"
GREY = "#5b6068"
NULL = "#d9dde3"
RULE = "#e3e6ea"

STYLE = f"""
text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
.tick {{ font-size: {26 * PT:.1f}px; fill: {GREY}; }}
.axis {{ font-size: {26 * PT:.1f}px; fill: {INK}; }}
.label {{ font-size: {31 * PT:.1f}px; font-weight: 700; }}
.detail {{ font-size: {25 * PT:.1f}px; fill: {GREY}; }}
.value {{ font-size: {29 * PT:.1f}px; font-weight: 700; }}
.legend {{ font-size: {26 * PT:.1f}px; fill: {INK}; }}
"""


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def page(svg: str, width: float, height: float, transparent: bool = False) -> str:
    background = "transparent" if transparent else "#ffffff"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"html,body{{margin:0;padding:0;background:{background};}} svg{{display:block;width:{width * 96:.2f}px;height:{height * 96:.2f}px;}}"
        "</style></head><body>" + svg + "</body></html>"
    )


def render(name: str, html: str, width: float, height: float, transparent: bool = False) -> Path:
    OUT.mkdir(exist_ok=True)
    source = OUT / f"{name}.html"
    source.write_text(html)
    target = OUT / f"{name}.png"
    command = [
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2.5",
        f"--window-size={round(width * 96)},{round(height * 96)}", f"--screenshot={target}", "--virtual-time-budget=5000",
    ]
    if transparent:
        command.append("--default-background-color=00000000")
    subprocess.run(command + [source.as_uri()], check=True, capture_output=True)
    return target


def equation_figure() -> Path:
    """The model for a poster audience: each formula line with a plain-language line under it. The variance of
    variant j's effect is s_j tau_j: tau_j from its class's learned mixing distribution g_c, s_j the scale its
    imputation quality (a known offset, r^2) and the learned smooths of allele frequency and annotations set."""
    width, height = 13.2, 6.4
    rows = [
        ("<i>y</i> = Σ<sub><i>j</i></sub> <i>x<sub>j</sub></i> β<sub><i>j</i></sub> + covariates + ε",
         "trait = genotypes × effects (SNVs, indels, SVs) + covariates + noise"),
        ("β<sub><i>j</i></sub> ~ N(0, <i>s<sub>j</sub></i> τ<sub><i>j</i></sub>)",
         "each variant’s effect follows a normal distribution with its own variance"),
        ("τ<sub><i>j</i></sub> ~ <i>g</i><sub>class</sub>",
         "each variant class (SNV, deletion, …) has its own variance distribution"),
        ("log <i>s<sub>j</sub></i> = log <i>r</i><sup>2</sup><sub style='margin-left:-0.35em'><i>j</i></sub> + <i>h</i>(freq.) + <i>f</i>(length) + ⋯",
         "the variance also depends on imputation quality (r²) and on smooth functions of allele frequency, SV length and other annotations"),
    ]
    body = "".join(
        f"<div class='row'><div class='math'>{math}</div><div class='plain'>{plain}</div></div>" for math, plain in rows
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;1,400&display=swap'>"
        "<style>html,body{margin:0;background:transparent;}"
        f".rows{{width:{width * 96:.1f}px;height:{height * 96:.1f}px;box-sizing:border-box;padding:0.08in 0.12in;display:flex;flex-direction:column;justify-content:space-between;}}"
        ".math{font-family:'STIX Two Text','Times New Roman',serif;font-size:46pt;line-height:1.2;color:#000;white-space:nowrap;}"
        f".plain{{font-family:Arial,Helvetica,sans-serif;font-size:27pt;line-height:1.25;color:#000;margin-top:2px;}}"
        "sub,sup{font-size:0.66em;line-height:0;}</style></head>"
        f"<body><div class='rows'>{body}</div></body></html>"
    )
    return render("equation", html, width, height, transparent=True)


# ------------------------------------------------------------------ the simulation study (simulation_summary.json)

AMBER = "#C8781E"
PURPLE = "#7D4E9E"
LIGHT = "#9aa3ad"
SIM_STYLE = STYLE + f"""
.facet {{ font-size: {28 * PT:.1f}px; font-weight: 700; fill: {INK}; }}
.key {{ font-size: {26 * PT:.1f}px; fill: {INK}; }}
.end {{ font-size: {22 * PT:.1f}px; font-weight: 700; }}
.row {{ font-size: {24 * PT:.1f}px; fill: {INK}; }}
.stat {{ font-size: {26 * PT:.1f}px; font-weight: 700; fill: {INK}; }}
"""
CHAR = 0.55
"""Mean Arial advance per em, for laying out labels."""


def text_width(text: str, size_pt: float) -> float:
    return len(text) * CHAR * size_pt * PT


def nice_range(low: float, high: float, steps=(0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0), most: int = 6):
    if low == high:
        low, high = low - steps[0], high + steps[0]
    for step in steps:
        if (high - low) / step <= most:
            break
    return math.floor(low / step + 1e-9) * step, math.ceil(high / step - 1e-9) * step, step


def tick_text(value: float, step: float, percent: bool = False) -> str:
    if percent:
        return f"{value * 100:.0f}%"
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return f"{value:.{decimals}f}".replace("-", "−")


def sizes(summary: dict, point: str) -> list[str]:
    return sorted(summary["points"][point]["sizes"], key=int)


def finite(values) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def thousands(count: int) -> str:
    return f"{count // 1000}k" if count % 1000 == 0 else f"{count:,}"


def spread_labels(wanted: list[float], gap: float, low: float, high: float) -> list[float]:
    """Label y positions as close to ``wanted`` as possible, at least ``gap`` apart, within [low, high]."""
    order = sorted(range(len(wanted)), key=lambda index: wanted[index])
    placed = [wanted[index] for index in order]
    if not placed:
        return []
    # push down from the top bound, then up from the bottom bound; only the ends are clamped, so the gaps survive
    placed[0] = max(placed[0], low)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    placed[-1] = min(placed[-1], high)
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = min(placed[index], placed[index + 1] - gap)
    if placed[0] < low:
        # more labels than the span holds at this gap: spread them evenly over it
        step = (high - low) / max(len(placed) - 1, 1)
        placed = [low + step * index for index in range(len(placed))]
    result = [0.0] * len(wanted)
    for slot, index in enumerate(order):
        result[index] = placed[slot]
    return result


ACCURACY_PANELS = (("null", "No causal SVs"), ("sv_5pct", "5% causal SVs"), ("sv_20pct", "20% causal SVs"))
ACCURACY_METHODS = (
    # (summary key, key label, colour, filled marker, horizontal offset)
    ("with_sv_auc", "SV-PGS with SVs", TEAL, True, 18.0),
    ("snv_indel_auc", "SV-PGS without SVs", NAVY, False, 6.0),
    ("ldpred2_allvar_auc", "LDpred2, all classes", AMBER, False, -6.0),
    ("ldpred2_auc", "LDpred2, SNV subset", AMBER, True, -18.0),
)
NO_VALUE = [math.nan, math.nan]


def auc_interval(row: dict, key: str) -> dict:
    """Read the collector's replicate interval; old summaries without one show only their measured mean."""
    return row.get("auc_intervals", {}).get(key, {"mean": row.get(key, NO_VALUE)[0], "lower": math.nan, "upper": math.nan})


def accuracy_figure(summary: dict) -> Path:
    """Test AUC against training size, one panel per causal-SV share: SV-PGS with and without SVs, LDpred2-auto on
    an SNV subset and every variant class at MAF >= 1%, and the analytic expected AUC (dashed)."""
    width, height = 16.70, 4.80
    w, h = width * 100, height * 100
    left, right, gap = 140.0, 30.0, 55.0
    # a method is keyed only where it has a value: the all-variant LDpred2 arm joined the grid after the others
    shown = [method for method in ACCURACY_METHODS if any(
        math.isfinite(row.get(method[0], NO_VALUE)[0]) for point, _title in ACCURACY_PANELS for row in summary["points"][point]["sizes"].values())]
    entries = [(label, colour, filled, False) for _key, label, colour, filled, _off in shown]
    # the key, centred over the panels: SV-PGS and the expected accuracy on one line, the LDpred2 arms on a second
    key_lines = [[entry for entry in entries if entry[0].startswith("SV-PGS")] + [("Expected", GREY, None, True)],
                 [entry for entry in entries if not entry[0].startswith("SV-PGS")]]
    key_lines = [line for line in key_lines if line]
    top, bottom = 132.0 + 46.0 * (len(key_lines) - 1), h - 115.0
    panel = (w - left - right - gap * (len(ACCURACY_PANELS) - 1)) / len(ACCURACY_PANELS)
    values = []
    for point, _title in ACCURACY_PANELS:
        for size in sizes(summary, point):
            row = summary["points"][point]["sizes"][size]
            for key, *_rest in ACCURACY_METHODS:
                entry = auc_interval(row, key)
                values += finite([entry["mean"], entry["lower"], entry["upper"]])
            values.append(row["expected_auc"])
    # tick labels at least 52 units (0.52 in) apart
    low, high, step = nice_range(min(values), max(values), steps=(0.005, 0.01, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2), most=int((bottom - top) // 52))
    counts = summary["train_counts"]

    def y(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{SIM_STYLE}</style>']
    for line_index, line in enumerate(key_lines):
        key_y = 40 + 46 * line_index
        span = sum(80 + text_width(label, 26) for label, *_rest in line) + 50 * (len(line) - 1)
        cursor = left + (w - left - right - span) / 2
        for label, colour, filled, dashed in line:
            if dashed:
                parts.append(f'<line x1="{cursor}" y1="{key_y}" x2="{cursor + 64}" y2="{key_y}" stroke="{GREY}" stroke-width="5" stroke-dasharray="13 9"/>')
            else:
                parts.append(f'<line x1="{cursor}" y1="{key_y}" x2="{cursor + 64}" y2="{key_y}" stroke="{colour}" stroke-width="6"/>')
                parts.append(f'<circle cx="{cursor + 32}" cy="{key_y}" r="11" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="5"/>')
            parts.append(f'<text class="key" x="{cursor + 80}" y="{key_y + 12}">{label}</text>')
            cursor += 80 + text_width(label, 26) + 50
    # y axis (shared)
    parts.append(f'<text class="axis" transform="translate(34 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Test AUC</text>')
    tick = low
    while tick <= high + 1e-9:
        parts.append(f'<text class="tick" x="{left - 16}" y="{y(tick) + 9:.1f}" text-anchor="end">{tick_text(tick, step)}</text>')
        tick += step
    for index, (point, title) in enumerate(ACCURACY_PANELS):
        x0 = left + index * (panel + gap)

        def x(count: int) -> float:
            return x0 + 34 + math.log2(count / counts[0]) / math.log2(counts[-1] / counts[0]) * (panel - 68)

        parts.append(f'<text class="facet" x="{x0}" y="{top - 22}">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{bottom}" x2="{x0 + panel}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
        parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
        tick = low
        while tick <= high + 1e-9:
            parts.append(f'<line x1="{x0 - 10}" y1="{y(tick):.1f}" x2="{x0}" y2="{y(tick):.1f}" stroke="{GREY}" stroke-width="3"/>')
            tick += step
        for count in counts:
            parts.append(f'<line x1="{x(count):.1f}" y1="{bottom}" x2="{x(count):.1f}" y2="{bottom + 10}" stroke="{GREY}" stroke-width="3"/>')
            parts.append(f'<text class="tick" x="{x(count):.1f}" y="{bottom + 44}" text-anchor="middle">{thousands(count)}</text>')
        if point not in summary["points"]:
            continue
        rows = summary["points"][point]["sizes"]
        present = sizes(summary, point)
        expected = " ".join(f"{x(int(size)):.1f},{y(rows[size]['expected_auc']):.1f}" for size in present)
        parts.append(f'<polyline points="{expected}" fill="none" stroke="{GREY}" stroke-width="5" stroke-dasharray="13 9"/>')
        for key, _label, colour, filled, offset in reversed(shown):
            points = [(x(int(size)) + offset, auc_interval(rows[size], key)) for size in present if math.isfinite(rows[size].get(key, NO_VALUE)[0])]
            if not points:
                continue
            parts.append('<polyline points="' + " ".join(f"{px:.1f},{y(entry['mean']):.1f}" for px, entry in points)
                         + f'" fill="none" stroke="{colour}" stroke-width="6" stroke-linejoin="round"/>')
            for px, entry in points:
                if math.isfinite(entry["lower"]):
                    parts.append(f'<line x1="{px:.1f}" y1="{y(entry["lower"]):.1f}" x2="{px:.1f}" y2="{y(entry["upper"]):.1f}" stroke="{colour}" stroke-width="4"/>')
                parts.append(f'<circle cx="{px:.1f}" cy="{y(entry["mean"]):.1f}" r="11" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="5"/>')
    parts.append(f'<text class="axis" x="{left + (w - left - right) / 2:.1f}" y="{h - 14}" text-anchor="middle">Training samples</text>')
    parts.append("</svg>")
    return render("ph_sim_accuracy", page("".join(parts), width, height), width, height)


POWER_SERIES = (
    ("sv_20pct_fold3", "20% SVs, 3× effect", NAVY),
    ("sv_20pct", "20% SVs", TEAL),
    ("sv_5pct_fold3", "5% SVs, 3× effect", PURPLE),
    ("sv_5pct", "5% SVs", AMBER),
    ("null", "No causal SVs", LIGHT),
)


def power_figure(summary: dict) -> Path:
    """Share of replicates whose one-sided paired DeLong test favours the SV arm at 0.05, against training size, per
    causal-SV scenario, with Wilson 95% intervals. No causal SV does not imply equal predictive accuracy: tagging and
    regularization can still change the two methods' scores."""
    width, height = 8.20, 4.40
    w, h = width * 100, height * 100
    left, top, bottom = 128.0, 28.0, h - 112.0
    plot_right = w - 12 - max(text_width(label, 22) for _point, label, _colour in POWER_SERIES) - 40
    counts = summary["train_counts"]

    def x(count: int) -> float:
        return left + 26 + math.log2(count / counts[0]) / math.log2(counts[-1] / counts[0]) * (plot_right - left - 52)

    def y(rate: float) -> float:
        return bottom - rate * (bottom - top)

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{SIM_STYLE}</style>']
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{plot_right}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left - 10}" y1="{y(rate):.1f}" x2="{left}" y2="{y(rate):.1f}" stroke="{GREY}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{left - 16}" y="{y(rate) + 9:.1f}" text-anchor="end">{rate * 100:.0f}%</text>')
    for count in counts:
        parts.append(f'<line x1="{x(count):.1f}" y1="{bottom}" x2="{x(count):.1f}" y2="{bottom + 10}" stroke="{GREY}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{x(count):.1f}" y="{bottom + 44}" text-anchor="middle">{thousands(count)}</text>')
    parts.append(f'<text class="axis" transform="translate(30 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Power</text>')
    parts.append(f'<text class="axis" x="{(left + plot_right) / 2:.1f}" y="{h - 14}" text-anchor="middle">Training samples</text>')
    ends, labels = [], []
    present = [entry for entry in POWER_SERIES if entry[0] in summary["points"]]
    for offset_index, (point, label, colour) in enumerate(present):
        offset = (offset_index - (len(present) - 1) / 2) * 9.0
        rows = summary["points"][point]["sizes"]
        points = [(x(int(size)) + offset, rows[size]["power"]) for size in sizes(summary, point) if rows[size]["power"]["replicates"]]
        if not points:
            continue
        parts.append('<polyline points="' + " ".join(f"{px:.1f},{y(row['rate']):.1f}" for px, row in points)
                     + f'" fill="none" stroke="{colour}" stroke-width="6" stroke-linejoin="round"/>')
        for px, row in points:
            parts.append(f'<line x1="{px:.1f}" y1="{y(row["lower"]):.1f}" x2="{px:.1f}" y2="{y(row["upper"]):.1f}" stroke="{colour}" stroke-width="3" opacity="0.55"/>')
            parts.append(f'<circle cx="{px:.1f}" cy="{y(row["rate"]):.1f}" r="9" fill="{colour}" stroke="#fff" stroke-width="3"/>')
        ends.append((points[-1][0], y(points[-1][1]["rate"]), colour))
        labels.append(label)
    placed = spread_labels([end[1] for end in ends], 34.0, top + 8, bottom - 4)
    for (end_x, end_y, colour), label, label_y in zip(ends, labels, placed):
        parts.append(f'<polyline points="{end_x + 14:.1f},{end_y:.1f} {plot_right + 10:.1f},{label_y:.1f} {plot_right + 24:.1f},{label_y:.1f}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        parts.append(f'<text class="end" x="{plot_right + 32:.1f}" y="{label_y + 9:.1f}" style="fill:{colour}">{label}</text>')
    parts.append("</svg>")
    return render("ph_sim_power", page("".join(parts), width, height), width, height)


GENOTYPE_CLASSES = (("SNV", "SNVs", NAVY), ("SV", "SVs", TEAL))
GENOTYPE_POINTS = ("null", "sv_20pct", "sv_20pct_fold3")


def genotypes_figure(summary: dict) -> Path:
    """Left: mean dosage r^2 of imputed SVs and SNVs against their true genotypes by minor allele frequency, with the
    middle half of variants shaded. Right: the AUC gain from adding SVs at the largest training size, per main scenario,
    on imputed genotypes (open) and on true genotypes (filled), with 95% t intervals over replicates."""
    width, height = 8.20, 4.40
    w, h = width * 100, height * 100
    accuracy = summary["genotype_accuracy"]
    by_frequency = accuracy["r2_by_frequency"]
    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{SIM_STYLE}</style>']
    top, bottom = 28.0, h - 128.0
    # left: r^2 against minor allele frequency on a log axis
    a0, a1 = 128.0, 372.0
    edges = by_frequency["edges"]
    frequency_low, frequency_high = math.log10(edges[0]), math.log10(edges[-1])

    def fx(frequency: float) -> float:
        return a0 + 12 + (math.log10(frequency) - frequency_low) / (frequency_high - frequency_low) * (a1 - a0 - 24)

    def fy(r2: float) -> float:
        return bottom - r2 * (bottom - top)

    parts.append(f'<line x1="{a0}" y1="{bottom}" x2="{a1}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
    parts.append(f'<line x1="{a0}" y1="{top}" x2="{a0}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
    for r2, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        parts.append(f'<line x1="{a0 - 10}" y1="{fy(r2):.1f}" x2="{a0}" y2="{fy(r2):.1f}" stroke="{GREY}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{a0 - 16}" y="{fy(r2) + 9:.1f}" text-anchor="end">{label}</text>')
    for frequency, label in ((0.001, "0.1%"), (0.01, "1%"), (0.1, "10%")):
        parts.append(f'<line x1="{fx(frequency):.1f}" y1="{bottom}" x2="{fx(frequency):.1f}" y2="{bottom + 10}" stroke="{GREY}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{fx(frequency):.1f}" y="{bottom + 42}" text-anchor="middle">{label}</text>')
    for name, label, colour in GENOTYPE_CLASSES:
        # each bin at its geometric midpoint
        bins = [(fx(math.sqrt(entry["low"] * entry["high"])), entry) for entry in by_frequency[name]["bins"] if entry["count"]]
        band = [f"{px:.1f},{fy(entry['q75']):.1f}" for px, entry in bins] + [f"{px:.1f},{fy(entry['q25']):.1f}" for px, entry in reversed(bins)]
        parts.append(f'<polygon points="{" ".join(band)}" fill="{colour}" fill-opacity="0.16"/>')
        parts.append('<polyline points="' + " ".join(f"{px:.1f},{fy(entry['mean']):.1f}" for px, entry in bins)
                     + f'" fill="none" stroke="{colour}" stroke-width="6" stroke-linejoin="round"/>')
        for px, entry in bins:
            parts.append(f'<circle cx="{px:.1f}" cy="{fy(entry["mean"]):.1f}" r="8" fill="{colour}" stroke="#fff" stroke-width="3"/>')
        # SNVs labelled under their first bins, where the SV band is low; SVs under their band's last three bins
        if name == "SNV":
            label_x, label_y, anchor = bins[0][0] - 8, fy(bins[0][1]["mean"]) + 46, "start"
        else:
            label_x, label_y, anchor = bins[-1][0] + 8, fy(min(entry["q25"] for _px, entry in bins[-3:])) + 38, "end"
        parts.append(f'<text class="end" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" style="fill:{colour}">{label}</text>')
    parts.append(f'<text class="axis" x="{(a0 + a1) / 2:.1f}" y="{h - 16}" text-anchor="middle">Allele frequency</text>')
    parts.append(f'<text class="axis" transform="translate(30 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Imputation r²</text>')
    # right: AUC gain from adding SVs, one row per scenario, imputed (open) above true (filled)
    b0, b1 = 448.0, 805.0
    largest = str(max(summary["train_counts"]))
    true_gains = accuracy.get("delta_auc", {})
    rows = []
    for point, label, colour in (entry for point in GENOTYPE_POINTS for entry in POWER_SERIES if entry[0] == point):
        imputed = summary["points"].get(point, {}).get("sizes", {}).get(largest, {}).get("delta_auc")
        true = true_gains.get(point, {}).get("true")
        rows.append((label, colour, [(entry, filled) for entry, filled in ((imputed, False), (true, True))
                                     if entry and math.isfinite(entry["mean"])]))
    values = [0.0] + finite(value for _label, _colour, marks in rows for entry, _filled in marks
                            for value in (entry["mean"], entry["lower"], entry["upper"]))
    low, high, step = nice_range(min(values), max(values), steps=(0.005, 0.01, 0.02, 0.025, 0.05, 0.1), most=4)

    def gx(value: float) -> float:
        return b0 + (value - low) / (high - low) * (b1 - b0)

    # the key in two lines at the top right, then the rows
    rows_top = 104.0
    key_x = b1 - 32 - text_width("True genotypes", 26)
    for line, (label, filled) in enumerate((("Imputed", False), ("True genotypes", True))):
        key_y = 34 + 40 * line
        parts.append(f'<circle cx="{key_x + 12:.1f}" cy="{key_y - 10:.1f}" r="10" fill="{GREY if filled else "#fff"}" stroke="{GREY}" stroke-width="5"/>')
        parts.append(f'<text class="key" x="{key_x + 32:.1f}" y="{key_y + 2:.1f}">{label}</text>')
    parts.append(f'<line x1="{b0}" y1="{bottom}" x2="{b1}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
    parts.append(f'<line x1="{gx(0.0):.1f}" y1="{rows_top}" x2="{gx(0.0):.1f}" y2="{bottom}" stroke="{GREY}" stroke-width="3" stroke-dasharray="10 8"/>')
    tick = low
    while tick <= high + 1e-9:
        parts.append(f'<line x1="{gx(tick):.1f}" y1="{bottom}" x2="{gx(tick):.1f}" y2="{bottom + 10}" stroke="{GREY}" stroke-width="3"/>')
        # the last label ends at the figure's edge instead of past it
        label = tick_text(tick, step)
        edge = gx(tick) + text_width(label, 26) / 2 > w - 4
        parts.append(f'<text class="tick" x="{w - 4 if edge else gx(tick):.1f}" y="{bottom + 42}" text-anchor="{"end" if edge else "middle"}">{label}</text>')
        tick += step
    row_height = (bottom - rows_top) / len(rows)
    for index, (label, colour, marks) in enumerate(rows):
        row_top = rows_top + index * row_height
        # a white outline breaks the zero line behind the label
        parts.append(f'<text class="row" x="{b0 + 6}" y="{row_top + 22:.1f}" style="fill:{colour}; font-weight:700; paint-order:stroke; '
                     f'stroke:#fff; stroke-width:10px; stroke-linejoin:round">{label}</text>')
        for entry, filled in marks:
            mark_y = row_top + (44 if not filled else 64)
            if math.isfinite(entry["lower"]):
                parts.append(f'<line x1="{gx(entry["lower"]):.1f}" y1="{mark_y:.1f}" x2="{gx(entry["upper"]):.1f}" y2="{mark_y:.1f}" stroke="{colour}" stroke-width="4"/>')
            parts.append(f'<circle cx="{gx(entry["mean"]):.1f}" cy="{mark_y:.1f}" r="10" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="5"/>')
    parts.append(f'<text class="axis" x="{(b0 + b1) / 2:.1f}" y="{h - 16}" text-anchor="middle">AUC gain from SVs</text>')
    parts.append("</svg>")
    return render("ph_sim_genotypes", page("".join(parts), width, height), width, height)


ARCHITECTURES = (
    # two blocks of the poster's figure: how many variants are causal and how many are SVs; SV effects and the trait
    (("null", "No causal SVs"), ("sv_5pct", "5% SVs"), ("sv_5pct_fold3", "5% SVs, 3× effect"), ("sv_20pct", "20% SVs"),
     ("sv_20pct_fold3", "20% SVs, 3× effect"), ("sparse_sv", "50 causal"), ("polygenic_sv", "2,000 causal")),
    (("length_positive", "Grows with length"), ("deletion_positive", "Deletions 3×"),
     ("multi_feature_positive", "Length, type, annotation"), ("sv_heavy_tail", "Heavy-tailed effects"),
     ("sv_two_point", "Two-point effects"), ("rare_disease", "Prevalence 1%"), ("low_heritability", "h² = 0.01")),
)
GAINS = (
    # (summary key, key label, colour, filled marker)
    ("delta_auc", "SV-PGS without SVs", TEAL, True),
    ("delta_auc_vs_ldpred2", "LDpred2, SNV subset", AMBER, True),
    ("delta_auc_vs_ldpred2_allvar", "LDpred2, all classes", AMBER, False),
)


def architectures_figure(summary: dict, width: float = 16.95, height: float = 4.6) -> Path:
    """For every simulated architecture, at its largest training size: the mean test AUC gain of SV-PGS with SVs over
    each other method, with its 95% interval over replicates, on one axis shared by both blocks of architectures."""
    blocks = [[(point, label) for point, label in block if point in summary["points"]] for block in ARCHITECTURES]
    blocks = [block for block in blocks if block]
    rows = [row for block in blocks for row in block]
    cells = {point: summary["points"][point]["sizes"][sizes(summary, point)[-1]] for point, _label in rows}
    gains = [gain for gain in GAINS if any(math.isfinite(cells[point].get(gain[0], {}).get("mean", math.nan)) for point, _label in rows)]
    w, h = width * 100, height * 100
    key_y, top, bottom = 40.0, 84.0, h - 112.0
    pitch = (bottom - top) / max(len(block) for block in blocks)
    # each block's row labels as wide as its longest; the panels share what is left
    labels = [24 + max(text_width(label, 24) for _point, label in block) for block in blocks]
    gap, right = 70.0, 40.0
    panel = (w - right - gap * (len(blocks) - 1) - sum(labels)) / len(blocks)
    values = [0.0]
    for point, _label in rows:
        for key, *_rest in gains:
            entry = cells[point].get(key, {})
            values += finite([entry.get("lower"), entry.get("upper"), entry.get("mean")])
    # tick labels at least 120 units apart
    low, high, step = nice_range(min(values), max(values), steps=(0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1), most=int(panel // 120))
    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{SIM_STYLE}</style>']
    spans = [40 + text_width(label, 26) for _key, label, _colour, _filled in gains]
    cursor = (w - sum(spans) - 50 * (len(spans) - 1)) / 2
    for (_key, label, colour, filled), span in zip(gains, spans):
        parts.append(f'<circle cx="{cursor + 12:.1f}" cy="{key_y}" r="11" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="5"/>')
        parts.append(f'<text class="key" x="{cursor + 34:.1f}" y="{key_y + 12}">{label}</text>')
        cursor += span + 50
    for block_index, block in enumerate(blocks):
        x0 = sum(labels[:block_index + 1]) + block_index * (panel + gap)

        def x(value: float) -> float:
            return x0 + (value - low) / (high - low) * panel

        parts.append(f'<line x1="{x(0.0):.1f}" y1="{top}" x2="{x(0.0):.1f}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
        parts.append(f'<line x1="{x0}" y1="{bottom}" x2="{x0 + panel:.1f}" y2="{bottom}" stroke="{GREY}" stroke-width="3"/>')
        tick = low
        while tick <= high + 1e-9:
            text = tick_text(tick, step)
            parts.append(f'<line x1="{x(tick):.1f}" y1="{bottom}" x2="{x(tick):.1f}" y2="{bottom + 10}" stroke="{GREY}" stroke-width="3"/>')
            parts.append(f'<text class="tick" x="{x(tick):.1f}" y="{bottom + 42}" text-anchor="middle">{text if abs(tick) > 1e-12 else "0"}</text>')
            tick += step
        for index, (point, label) in enumerate(block):
            cy = top + pitch * index + pitch / 2
            parts.append(f'<text class="row" x="{x0 - 16:.1f}" y="{cy + 9:.1f}" text-anchor="end">{label}</text>')
            for gain_index, (key, _label, colour, filled) in enumerate(gains):
                entry = cells[point].get(key, {})
                if not math.isfinite(entry.get("mean", math.nan)):
                    continue
                y = cy + (gain_index - (len(gains) - 1) / 2) * 13.0
                if math.isfinite(entry["lower"]):
                    parts.append(f'<line x1="{x(entry["lower"]):.1f}" y1="{y:.1f}" x2="{x(entry["upper"]):.1f}" y2="{y:.1f}" stroke="{colour}" stroke-width="4" stroke-linecap="round"/>')
                parts.append(f'<circle cx="{x(entry["mean"]):.1f}" cy="{y:.1f}" r="9" fill="{colour if filled else "#fff"}" stroke="{colour}" stroke-width="4"/>')
    counts = sorted({int(sizes(summary, point)[-1]) for point, _label in rows})
    parts.append(f'<text class="axis" x="{w / 2:.1f}" y="{h - 16}" text-anchor="middle">Test AUC gain of SV-PGS with SVs at {" / ".join(thousands(count) for count in counts)} training samples</text>')
    parts.append("</svg>")
    return render("fig_sim_architectures", page("".join(parts), width, height), width, height)


def simulation_figures(summary: dict | None) -> list[Path]:
    """Each simulation figure whose grid points have results; a placeholder for the others."""
    made = []
    points = summary["points"] if summary else {}
    made.append(accuracy_figure(summary) if all(point in points for point, _title in ACCURACY_PANELS) else placeholder_figure("ph_sim_accuracy"))
    made.append(power_figure(summary) if "null" in points and any(point in points for point, *_rest in POWER_SERIES[:-1]) else placeholder_figure("ph_sim_power"))
    made.append(genotypes_figure(summary) if summary and "r2_by_frequency" in summary.get("genotype_accuracy", {})
                else placeholder_figure("ph_sim_genotypes"))
    if summary and len(points) > 1:
        made.append(architectures_figure(summary))
    else:
        # The builder includes this optional panel by filename: never reuse a previous run's result.
        for suffix in (".png", ".html"):
            (OUT / f"fig_sim_architectures{suffix}").unlink(missing_ok=True)
    return made


# ------------------------------------------------------------------ placeholders for results not yet run

PLACEHOLDERS = {
    # name: (width in, height in, source, title, what the figure will show)
    "ph_sim_accuracy": (16.70, 4.80, "simulation", "Test AUC vs. training size",
                        "SV-PGS with and without SVs, and LDpred2-auto, at 0%, 5% and 20% causal SVs, with the expected accuracy"),
    "ph_sim_power": (8.20, 4.40, "simulation", "Power vs. training size",
                     "Share of replicates in which adding SVs raised AUC (one-sided paired DeLong test, α = 0.05)"),
    "ph_sim_genotypes": (8.20, 4.40, "simulation", "SV genotype accuracy",
                         "Imputation r² of SVs and SNVs by allele frequency, and the AUC gain from SVs on imputed vs. true genotypes"),
    "ph_aou_hypertension": (15.40, 5.00, "All of Us", "SV effects on hypertension",
                            "Estimated SV effects by SV type, and |effect| vs. SV length"),
    "ph_aou_diseases": (15.40, 4.60, "All of Us", "Test AUC by disease",
                        "SNV-only and SNV + SV models, and LDpred2-auto, for each disease"),
}


def placeholder_figure(name: str) -> Path:
    width, height, source, title, detail = PLACEHOLDERS[name]
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"html,body{{margin:0;background:#fff;}} .box{{box-sizing:border-box;width:{width * 96:.1f}px;height:{height * 96:.1f}px;"
        f"border:6px dashed #9aa3ad;border-radius:18px;background:#eef1f4;display:flex;flex-direction:column;align-items:center;"
        f"justify-content:center;gap:18px;padding:30px 50px;text-align:center;font-family:Arial,Helvetica,sans-serif;color:{INK};}}"
        f".tag{{font-size:24pt;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};}}"
        f".title{{font-size:38pt;font-weight:700;line-height:1.15;color:{NAVY};}}"
        f".detail{{font-size:26pt;line-height:1.3;color:{GREY};max-width:92%;}}"
        "</style></head><body><div class='box'>"
        f"<div class='tag'>Placeholder: {source} result</div><div class='title'>{title}</div><div class='detail'>{detail}</div>"
        "</div></body></html>"
    )
    return render(name, html, width, height)


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "simulation_summary.json"
    if len(sys.argv) > 2:
        OUT = Path(sys.argv[2])
    summary = json.loads(source.read_text()) if source.is_file() else None
    aou = [name for name in PLACEHOLDERS if name.startswith("ph_aou")]
    for made in (equation_figure(), *simulation_figures(summary), *(placeholder_figure(name) for name in aou)):
        print(made)
