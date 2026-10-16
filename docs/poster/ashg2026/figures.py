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


# ------------------------------------------------------------------ SV type and expression (MAGE), first column

SKEW = [
    ("Deletion overlapping an exon", "78 SVs, 42 genes", RED, -1.33, (-0.28, 0.25), (-2.52, -0.21)),
    ("Insertion or duplication overlapping an exon", "93 SVs, 75 genes", NAVY, 0.66, (-0.22, 0.21), (0.25, 1.30)),
]


def skew_figure() -> Path:
    width, height = 13.4, 5.8
    w, h = width * 100, height * 100
    plot_left, plot_right = 40.0, w - 40.0
    low, high = -2.75, 1.5
    top = 130.0
    row = 165.0
    bottom = top + row * (len(SKEW) - 1) + 125 + 52

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg"><style>{STYLE}</style>']
    # legend
    parts.append(f'<rect x="{plot_left}" y="40" width="70" height="36" rx="5" fill="{NULL}"/>')
    parts.append(f'<text class="legend" x="{plot_left + 86}" y="70">permutation 95% range</text>')
    parts.append(f'<line x1="{plot_left + 560}" y1="58" x2="{plot_left + 640}" y2="58" stroke="{INK}" stroke-width="7" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{plot_left + 600}" cy="58" r="15" fill="{INK}" stroke="#fff" stroke-width="5"/>')
    parts.append(f'<text class="legend" x="{plot_left + 658}" y="70">mean z, 95% CI</text>')
    for tick in (-2, -1, 0, 1):
        tx = x(tick)
        stroke, weight = (GREY, 4) if tick == 0 else (RULE, 3)
        # gridlines run inside each row's band only, never through the row labels
        for index in range(len(SKEW)):
            cy = top + row * index + 105
            parts.append(f'<line x1="{tx:.1f}" y1="{cy - 52:.1f}" x2="{tx:.1f}" y2="{cy + 52:.1f}" stroke="{stroke}" stroke-width="{weight}"/>')
        label = "0" if tick == 0 else f"{tick:+d}".replace("-", "−")
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{bottom + 24}" text-anchor="middle">{label}</text>')
    parts.append(f'<text class="axis" x="{(plot_left + plot_right) / 2:.1f}" y="{h - 30}" text-anchor="middle">Mean z-score for association with expression of the overlapped gene</text>')
    for index, (label, detail, colour, mean, null, ci) in enumerate(SKEW):
        base = top + row * index
        parts.append(f'<text class="label" x="{plot_left}" y="{base + 18:.1f}">{label} <tspan class="detail" dx="10">{detail}</tspan></text>')
        cy = base + 105
        parts.append(f'<rect x="{x(null[0]):.1f}" y="{cy - 30:.1f}" width="{x(null[1]) - x(null[0]):.1f}" height="60" rx="6" fill="{NULL}"/>')
        parts.append(f'<line x1="{x(ci[0]):.1f}" y1="{cy:.1f}" x2="{x(ci[1]):.1f}" y2="{cy:.1f}" stroke="{colour}" stroke-width="9" stroke-linecap="round"/>')
        for end in ci:
            parts.append(f'<line x1="{x(end):.1f}" y1="{cy - 22:.1f}" x2="{x(end):.1f}" y2="{cy + 22:.1f}" stroke="{colour}" stroke-width="9" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x(mean):.1f}" cy="{cy:.1f}" r="22" fill="{colour}" stroke="#fff" stroke-width="7"/>')
    parts.append("</svg>")
    return render("fig_sign_skew", page("".join(parts), width, height), width, height)


# ------------------------------------------------------------------ the model equation, transparent, for the Methods panel

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


# ------------------------------------------------------------------ placeholders for results not yet run

PLACEHOLDERS = {
    # name: (width in, height in, source, title, what the figure will show)
    "ph_sim_accuracy": (16.70, 5.30, "simulation", "Held-out AUC vs. training size",
                        "SV-PGS with and without SVs, and LDpred2-auto, at 0%, 5% and 20% causal SVs, with the expected accuracy"),
    "ph_sim_power": (8.20, 4.80, "simulation", "Power vs. training size",
                     "Share of replicates in which adding SVs raised AUC (one-sided paired DeLong test, α = 0.05)"),
    "ph_sim_effects": (8.20, 4.80, "simulation", "SV effect sizes",
                       "Estimated vs. simulated SV effects, and |effect| vs. SV length"),
    "ph_aou_hypertension": (15.40, 5.00, "All of Us", "SV effects on hypertension",
                            "Estimated SV effects by SV type, and |effect| vs. SV length"),
    "ph_aou_diseases": (15.40, 4.60, "All of Us", "Held-out AUC by disease",
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
    for made in (skew_figure(), equation_figure(), *(placeholder_figure(name) for name in PLACEHOLDERS)):
        print(made)
