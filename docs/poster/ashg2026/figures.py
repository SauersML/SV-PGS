"""The science poster's figures and its model equation, as print-resolution PNGs for the PowerPoint template.

Every value is the repository's recorded measurement, as in build_poster.py:
- SV sign skew: docs/design/LOCATION_SCALE.md [real: bench-real genes500, MAGE + 1kGP]
- bench-real baselines and SV credit: docs/design/EVALUATION.md [real]

Each figure is drawn in an SVG whose units are hundredths of an inch at the figure's printed size (the template's picture
frame), so 1 pt of type is 100 / 72 units. Chrome renders it at 2.5x (240 dpi).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PT = 100.0 / 72.0

NAVY = "#235078"
TEAL = "#1482A5"
RED = "#C0392B"
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


# ------------------------------------------------------------------ SV sign skew (the large figure, 15.64 x 11.76 in)

SKEW = [
    ("Deletion overlapping an exon", "78 SVs, 42 genes", RED, -1.33, (-0.28, 0.25), (-2.52, -0.21)),
    ("Insertion/duplication overlapping an exon", "93 SVs, 75 genes", NAVY, 0.66, (-0.22, 0.21), (0.25, 1.30)),
    ("Deletion in the gene body, no exon", "866 SVs", RED, 0.05, (-0.07, 0.08), (-0.15, 0.48)),
    ("Insertion/duplication in the gene body, no exon", "737 SVs", NAVY, 0.01, (-0.08, 0.08), (-0.47, 0.06)),
]


def skew_figure() -> Path:
    width, height = 15.64, 11.76
    w, h = width * 100, height * 100
    plot_left, plot_right = 70.0, w - 60.0
    low, high = -2.75, 1.5
    top = 160.0
    row = 222.0
    bottom = top + row * (len(SKEW) - 1) + 125 + 52

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{STYLE}</style>']
    # legend
    parts.append(f'<rect x="{plot_left}" y="40" width="70" height="36" rx="5" fill="{NULL}"/>')
    parts.append(f'<text class="legend" x="{plot_left + 86}" y="70">permutation null, 95%</text>')
    parts.append(f'<line x1="{plot_left + 830}" y1="58" x2="{plot_left + 910}" y2="58" stroke="{INK}" stroke-width="7" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{plot_left + 870}" cy="58" r="15" fill="{INK}" stroke="#fff" stroke-width="5"/>')
    parts.append(f'<text class="legend" x="{plot_left + 928}" y="70">mean z, gene-bootstrap 95% CI</text>')
    for tick in (-2, -1, 0, 1):
        tx = x(tick)
        stroke, weight = (GREY, 4) if tick == 0 else (RULE, 3)
        # gridlines run inside each row's band only, never through the row labels
        for index in range(len(SKEW)):
            cy = top + row * index + 125
            parts.append(f'<line x1="{tx:.1f}" y1="{cy - 52:.1f}" x2="{tx:.1f}" y2="{cy + 52:.1f}" stroke="{stroke}" stroke-width="{weight}"/>')
        label = "0" if tick == 0 else f"{tick:+d}".replace("-", "−")
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{bottom + 44}" text-anchor="middle">{label}</text>')
    parts.append(f'<text class="axis" x="{(plot_left + plot_right) / 2:.1f}" y="{h - 40}" text-anchor="middle">Mean association z of each SV with its own gene’s expression</text>')
    for index, (label, detail, colour, mean, null, ci) in enumerate(SKEW):
        base = top + row * index
        parts.append(f'<text class="label" x="{plot_left}" y="{base + 18:.1f}">{label} <tspan class="detail" dx="10">{detail}</tspan></text>')
        cy = base + 125
        parts.append(f'<rect x="{x(null[0]):.1f}" y="{cy - 30:.1f}" width="{x(null[1]) - x(null[0]):.1f}" height="60" rx="6" fill="{NULL}"/>')
        parts.append(f'<line x1="{x(ci[0]):.1f}" y1="{cy:.1f}" x2="{x(ci[1]):.1f}" y2="{cy:.1f}" stroke="{colour}" stroke-width="9" stroke-linecap="round"/>')
        for end in ci:
            parts.append(f'<line x1="{x(end):.1f}" y1="{cy - 22:.1f}" x2="{x(end):.1f}" y2="{cy + 22:.1f}" stroke="{colour}" stroke-width="9" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x(mean):.1f}" cy="{cy:.1f}" r="22" fill="{colour}" stroke="#fff" stroke-width="7"/>')
        if index < 2:
            value = f"{mean:+.2f}".replace("-", "−")
            parts.append(f'<text class="value" x="{x(mean):.1f}" y="{cy - 38:.1f}" text-anchor="middle">mean z = {value}</text>')
    parts.append("</svg>")
    return render("fig_sign_skew", page("".join(parts), width, height), width, height)


# ------------------------------------------------------------------ bench-real baselines (10.94 x 8.40 in)

BASELINES = [
    ("mr.ash", (0.071, 0.081), 0.023),
    ("Lead variant", (0.061, 0.068), 0.022),
    ("GBLUP", (0.047, 0.053), 0.013),
]


def baseline_figure() -> Path:
    width, height = 10.94, 8.40
    w, h = width * 100, height * 100
    label_right = 300.0
    plot_left, plot_right = 340.0, w - 50.0
    low, high = 0.0, 0.09
    top = 190.0
    row = 165.0
    bottom = top + row * len(BASELINES)

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{STYLE}</style>']
    parts.append(f'<rect x="{plot_left}" y="44" width="80" height="30" rx="15" fill="{NAVY}"/>')
    parts.append(f'<text class="legend" x="{plot_left + 96}" y="70">other held-out superpopulations</text>')
    parts.append(f'<circle cx="{plot_left + 18}" cy="113" r="16" fill="#fff" stroke="{TEAL}" stroke-width="7"/>')
    parts.append(f'<text class="legend" x="{plot_left + 96}" y="124">African ancestry held out</text>')
    for tick in (0.0, 0.02, 0.04, 0.06, 0.08):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top - 30}" x2="{tx:.1f}" y2="{bottom - 10}" stroke="{RULE}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{bottom + 34}" text-anchor="middle">{fmt(tick) if tick else "0"}</text>')
    parts.append(f'<text class="axis" x="{(plot_left + plot_right) / 2:.1f}" y="{h - 34}" text-anchor="middle">Mean held-out r² per gene</text>')
    for index, (method, (lo, hi), afr) in enumerate(BASELINES):
        cy = top + row * index + row / 2 - 10
        parts.append(f'<text class="label" x="{label_right}" y="{cy + 12:.1f}" text-anchor="end">{method}</text>')
        parts.append(f'<rect x="{x(lo) - 18:.1f}" y="{cy - 18:.1f}" width="{x(hi) - x(lo) + 36:.1f}" height="36" rx="18" fill="{NAVY}"/>')
        parts.append(f'<text class="value" x="{x(hi) + 18:.1f}" y="{cy - 34:.1f}" text-anchor="end">{lo:.3f}–{hi:.3f}</text>')
        parts.append(f'<circle cx="{x(afr):.1f}" cy="{cy:.1f}" r="17" fill="#fff" stroke="{TEAL}" stroke-width="8"/>')
        parts.append(f'<text class="value" x="{x(afr) + 32:.1f}" y="{cy + 12:.1f}">{afr:.3f}</text>')
    parts.append("</svg>")
    return render("fig_baselines", page("".join(parts), width, height), width, height)


# ------------------------------------------------------------------ SV credit (6.96 x 8.40 in), vertical

CREDIT = [("mr.ash", (1.9, 3.9)), ("GBLUP", (0.2, 0.3))]
COLUMN_SHARE = 0.32


def credit_figure() -> Path:
    width, height = 6.96, 8.40
    w, h = width * 100, height * 100
    plot_left, plot_right = 130.0, w - 40.0
    plot_top, plot_bottom = 70.0, h - 130.0
    low, high = 0.0, 4.2

    def y(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{STYLE}</style>']
    for tick in (0, 1, 2, 3, 4):
        ty = y(tick)
        parts.append(f'<line x1="{plot_left}" y1="{ty:.1f}" x2="{plot_right}" y2="{ty:.1f}" stroke="{RULE}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{plot_left - 18}" y="{ty + 10:.1f}" text-anchor="end">{tick}%</text>')
    parts.append(
        f'<text class="axis" transform="translate(36 {(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">'
        f"SV share of held-out prediction</text>"
    )
    share = y(COLUMN_SHARE)
    parts.append(f'<line x1="{plot_left}" y1="{share:.1f}" x2="{plot_right}" y2="{share:.1f}" stroke="{RED}" stroke-width="5" stroke-dasharray="14 10"/>')
    parts.append(f'<text class="legend" x="{plot_left + 12}" y="{share + 42:.1f}" style="fill:{RED};font-weight:700;font-size:{21 * PT:.1f}px">SVs: 0.32% of variants</text>')
    slot = (plot_right - plot_left) / len(CREDIT)
    for index, (method, (lo, hi)) in enumerate(CREDIT):
        cx = plot_left + slot * (index + 0.5)
        parts.append(f'<rect x="{cx - 22:.1f}" y="{y(hi) - 22:.1f}" width="44" height="{y(lo) - y(hi) + 44:.1f}" rx="22" fill="{NAVY}"/>')
        parts.append(f'<text class="value" x="{cx:.1f}" y="{y(hi) - 40:.1f}" text-anchor="middle">{fmt(lo)}–{fmt(hi)}%</text>')
        parts.append(f'<text class="label" x="{cx:.1f}" y="{plot_bottom + 50:.1f}" text-anchor="middle">{method}</text>')
    parts.append("</svg>")
    return render("fig_sv_credit", page("".join(parts), width, height), width, height)


# ------------------------------------------------------------------ the model equation, transparent, for the Methods panel

def equation_figure() -> Path:
    """The model for a poster audience: each formula line with a plain-language line under it. The variance of
    variant j's effect is s_j tau_j: tau_j from its class's learned mixing distribution g_c, s_j the scale its
    imputation quality (a known offset, r^2) and the learned smooths of allele frequency and annotations set."""
    width, height = 11.6, 7.0
    rows = [
        ("<i>y</i> = Σ<sub><i>j</i></sub> <i>x<sub>j</sub></i> β<sub><i>j</i></sub> + covariates + ε",
         "trait = genotypes × effects of SNVs, indels and SVs, plus covariates and noise"),
        ("β<sub><i>j</i></sub> ~ N(0, <i>s<sub>j</sub></i> τ<sub><i>j</i></sub>)",
         "each variant’s effect is normal, with a variance of its own"),
        ("τ<sub><i>j</i></sub> ~ <i>g</i><sub>class</sub>",
         "drawn from a smooth distribution learned for each variant class (SNV, deletion, insertion, …)"),
        ("log <i>s<sub>j</sub></i> = log <i>r</i><sup>2</sup><sub style='margin-left:-0.35em'><i>j</i></sub> + <i>h</i>(freq.) + <i>f</i>(length) + ⋯",
         "scaled by imputation quality r², and learned smooth functions of allele frequency, SV length and other annotations"),
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
        f".plain{{font-family:Arial,Helvetica,sans-serif;font-size:29pt;line-height:1.25;color:{GREY};margin-top:2px;}}"
        "sub,sup{font-size:0.66em;line-height:0;}</style></head>"
        f"<body><div class='rows'>{body}</div></body></html>"
    )
    return render("equation", html, width, height, transparent=True)


# ------------------------------------------------------------------ placeholders for the All of Us results, not yet run

PLACEHOLDERS = {
    # name: (width in, height in, title, what the figure will show)
    "ph_hypertension_effects": (10.94, 8.40, "Hypertension: SV effect sizes",
                                "Posterior effect of each SV on hypertension risk, risk-increasing against risk-decreasing, by SV type"),
    "ph_effect_by_length": (6.96, 8.40, "Effect size by SV length",
                            "Absolute posterior effect against SV length, deletions and insertions apart"),
    "ph_fifteen_diseases": (18.20, 9.40, "15 diseases: SNV-only against SNV + SV",
                            "Held-out AUC and log loss of the two models per disease, paired on the same people and folds"),
}


def placeholder_figure(name: str) -> Path:
    width, height, title, detail = PLACEHOLDERS[name]
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"html,body{{margin:0;background:#fff;}} .box{{box-sizing:border-box;width:{width * 96:.1f}px;height:{height * 96:.1f}px;"
        f"border:6px dashed #9aa3ad;border-radius:18px;background:#eef1f4;display:flex;flex-direction:column;align-items:center;"
        f"justify-content:center;gap:22px;padding:40px 60px;text-align:center;font-family:Arial,Helvetica,sans-serif;color:{INK};}}"
        f".tag{{font-size:26pt;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{TEAL};}}"
        f".title{{font-size:40pt;font-weight:700;line-height:1.15;color:{NAVY};}}"
        f".detail{{font-size:27pt;line-height:1.3;color:{GREY};max-width:90%;}}"
        "</style></head><body><div class='box'>"
        f"<div class='tag'>All of Us result · placeholder</div><div class='title'>{title}</div><div class='detail'>{detail}</div>"
        "</div></body></html>"
    )
    return render(name, html, width, height)


if __name__ == "__main__":
    for made in (skew_figure(), baseline_figure(), credit_figure(), equation_figure(), *(placeholder_figure(name) for name in PLACEHOLDERS)):
        print(made)
