"""Builds the ASHG 2026 SV-PGS poster (sv-pgs-poster.html) from recorded project measurements.

Every number drawn here is copied from the repository's design documents, with the evidence tag they carry there:
- the SV sign skew: docs/design/LOCATION_SCALE.md [real]
- the bench-real baselines and SV credit: docs/design/EVALUATION.md, "Neutral benchmark results so far" [real]
- the chr1 ideogram: UCSC hg38 cytoBand.txt (GRCh38 G-banding), its chr1 rows in cytoBand_chr1.txt

Run `python3 build_poster.py`: it writes the screen page (sv-pgs-poster.html, fonts from Google Fonts) and the print page
(sv-pgs-poster-print.html, fonts inlined so a PDF cannot fall back; the font files are cached in fonts/, untracked). Print
the PDF with headless Chrome:

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --virtual-time-budget=25000 \
        --no-pdf-header-footer --print-to-pdf=SV-PGS_ASHG2026_poster_72x36in.pdf "file://$PWD/sv-pgs-poster-print.html"
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "sv-pgs-poster.html"


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ the chr1 ideogram

BAND_FILL = {
    "gneg": "#f4f5f7",
    "gpos25": "#c9ced6",
    "gpos50": "#9aa2b0",
    "gpos75": "#6c7586",
    "gpos100": "#3a4252",
    "gvar": "#b4bac4",
    "acen": "#c4504c",
}


def ideogram() -> str:
    bands = []
    for line in (HERE / "cytoBand_chr1.txt").read_text().splitlines():
        chrom, start, end, name, stain = line.split("\t")
        if chrom == "chr1":
            bands.append((int(start), int(end), name, stain))
    length = bands[-1][1]
    width, height = 6960.0, 44.0
    scale = width / length
    acen = [band for band in bands if band[3] == "acen"]
    left, right = acen[0][0] * scale, acen[-1][1] * scale
    middle = acen[0][1] * scale
    radius = height / 2
    pinch = height * 0.28
    outline = (
        f"M{radius},0 H{left} L{middle},{pinch} L{right},0 H{width - radius} "
        f"A{radius},{radius} 0 0 1 {width - radius},{height} H{right} L{middle},{height - pinch} L{left},{height} H{radius} "
        f"A{radius},{radius} 0 0 1 {radius},0 Z"
    )
    rects = "".join(
        f'<rect x="{start * scale:.2f}" y="0" width="{(end - start) * scale:.2f}" height="{height}" fill="{BAND_FILL[stain]}">'
        f"<title>1{name} ({stain})</title></rect>"
        for start, end, name, stain in bands
    )
    return (
        f'<svg class="ideogram" viewBox="-2 -2 {width + 4} {height + 4}" preserveAspectRatio="none" role="img" '
        f'aria-label="Human chromosome 1, GRCh38 G-banding">'
        f'<defs><clipPath id="chr1-clip"><path d="{outline}"/></clipPath></defs>'
        f'<g clip-path="url(#chr1-clip)">{rects}</g>'
        f'<path d="{outline}" fill="none" stroke="#151a22" stroke-width="2" vector-effect="non-scaling-stroke"/>'
        f"</svg>"
    )


# ------------------------------------------------------------------ figure 1: SV sign skew [real]

SKEW_ROWS = [
    # label, detail, colour token, observed mean z, permutation null 95%, gene-bootstrap 95% CI
    ("DEL × exon", "78 SVs · 42 genes", "loss", -1.33, (-0.28, 0.25), (-2.52, -0.21)),
    ("INS/DUP × exon", "93 SVs · 75 genes", "gain", 0.66, (-0.22, 0.21), (0.25, 1.30)),
    ("DEL in gene body", "866 SVs · no exon", "loss", 0.05, (-0.07, 0.08), (-0.15, 0.48)),
    ("INS/DUP in body", "737 SVs · no exon", "gain", 0.01, (-0.08, 0.08), (-0.47, 0.06)),
]


def skew_chart() -> str:
    width = 1000.0
    label_right = 330.0
    plot_left, plot_right = 372.0, 980.0
    low, high = -2.6, 1.4
    top = 40.0
    row = 128.0
    height = top + row * len(SKEW_ROWS) + 76.0

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Mean association z of SVs with their own gene expression by class and exon overlap">'
    ]
    # grid and zero line
    for tick in (-2, -1, 0, 1):
        tx = x(tick)
        stroke = "var(--ink-2)" if tick == 0 else "var(--rule)"
        weight = 3 if tick == 0 else 2
        parts.append(f'<line x1="{tx:.1f}" y1="{top - 18}" x2="{tx:.1f}" y2="{top + row * len(SKEW_ROWS) - 10}" stroke="{stroke}" stroke-width="{weight}"/>')
        label = "0" if tick == 0 else f"{tick:+d}".replace("-", "−")
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{top + row * len(SKEW_ROWS) + 30}" text-anchor="middle">{label}</text>')
    parts.append(
        f'<text class="axis" x="{(plot_left + plot_right) / 2:.1f}" y="{height - 6}" text-anchor="middle">'
        f"mean association z with the SV's own gene</text>"
    )
    for index, (label, detail, colour, mean, null, ci) in enumerate(SKEW_ROWS):
        cy = top + row * index + row / 2 - 12
        parts.append(f'<text class="row-label" x="{label_right}" y="{cy - 4:.1f}" text-anchor="end">{label}</text>')
        parts.append(f'<text class="row-detail" x="{label_right}" y="{cy + 30:.1f}" text-anchor="end">{detail}</text>')
        parts.append(
            f'<rect x="{x(null[0]):.1f}" y="{cy - 20:.1f}" width="{x(null[1]) - x(null[0]):.1f}" height="40" rx="4" fill="var(--null)">'
            f"<title>Permutation null 95%: {fmt(null[0])} to {fmt(null[1])}</title></rect>"
        )
        parts.append(
            f'<line x1="{x(ci[0]):.1f}" y1="{cy:.1f}" x2="{x(ci[1]):.1f}" y2="{cy:.1f}" stroke="var(--{colour})" stroke-width="5" stroke-linecap="round">'
            f"<title>Gene-bootstrap 95% CI: {fmt(ci[0])} to {fmt(ci[1])}</title></line>"
        )
        for end in ci:
            parts.append(f'<line x1="{x(end):.1f}" y1="{cy - 14:.1f}" x2="{x(end):.1f}" y2="{cy + 14:.1f}" stroke="var(--{colour})" stroke-width="5" stroke-linecap="round"/>')
        parts.append(
            f'<circle cx="{x(mean):.1f}" cy="{cy:.1f}" r="14" fill="var(--{colour})" stroke="var(--paper)" stroke-width="5">'
            f"<title>{label}: mean z {fmt(mean)}</title></circle>"
        )
        if index < 2:
            value = f"{mean:+.2f}".replace("-", "−")
            parts.append(f'<text class="value" x="{x(mean):.1f}" y="{cy - 30:.1f}" text-anchor="middle">{value}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ figure 3: bench-real baselines [real]

BASELINES = [
    # method, held-out r² range over the other held-out superpopulations, AFR held out
    ("mr.ash", (0.071, 0.081), 0.023),
    ("Lead variant", (0.061, 0.068), 0.022),
    ("GBLUP", (0.047, 0.053), 0.013),
]


def baseline_chart() -> str:
    width = 1000.0
    label_right = 250.0
    plot_left, plot_right = 290.0, 960.0
    low, high = 0.0, 0.09
    top = 52.0
    row = 116.0
    height = top + row * len(BASELINES) + 72.0

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    parts = [f'<svg class="chart" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Held-out r squared of three baselines on bench-real">']
    for tick in (0.0, 0.02, 0.04, 0.06, 0.08):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top - 14}" x2="{tx:.1f}" y2="{top + row * len(BASELINES) - 14}" stroke="var(--rule)" stroke-width="2"/>')
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{top + row * len(BASELINES) + 22}" text-anchor="middle">{fmt(tick) if tick else "0"}</text>')
    parts.append(f'<text class="axis" x="{(plot_left + plot_right) / 2:.1f}" y="{height - 6}" text-anchor="middle">mean held-out r² per gene</text>')
    for index, (method, (lo, hi), afr) in enumerate(BASELINES):
        cy = top + row * index + row / 2
        parts.append(f'<text class="row-label" x="{label_right}" y="{cy + 11:.1f}" text-anchor="end">{method}</text>')
        parts.append(
            f'<rect x="{x(lo) - 13:.1f}" y="{cy - 13:.1f}" width="{x(hi) - x(lo) + 26:.1f}" height="26" rx="13" fill="var(--ink)">'
            f"<title>{method}: {lo:.3f} to {hi:.3f}</title></rect>"
        )
        parts.append(f'<text class="value" x="{(x(lo) + x(hi)) / 2:.1f}" y="{cy - 26:.1f}" text-anchor="middle">{lo:.3f}–{hi:.3f}</text>')
        parts.append(
            f'<circle cx="{x(afr):.1f}" cy="{cy:.1f}" r="12" fill="var(--paper)" stroke="var(--ink)" stroke-width="5">'
            f"<title>{method}, AFR held out: {afr:.3f}</title></circle>"
        )
        parts.append(f'<text class="value muted" x="{x(afr) + 24:.1f}" y="{cy + 10:.1f}">{afr:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ figure 4: SV credit [real]

CREDIT = [
    ("mr.ash", (1.9, 3.9)),
    ("GBLUP", (0.2, 0.3)),
]
COLUMN_SHARE = 0.32


def credit_chart() -> str:
    width = 1000.0
    label_right = 250.0
    plot_left, plot_right = 290.0, 960.0
    low, high = 0.0, 4.2
    top = 112.0
    row = 116.0
    height = top + row * len(CREDIT) + 72.0

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    parts = [f'<svg class="chart" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="SV share of held-out predictive covariance against their share of columns">']
    for tick in (0, 1, 2, 3, 4):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top - 14}" x2="{tx:.1f}" y2="{top + row * len(CREDIT) - 14}" stroke="var(--rule)" stroke-width="2"/>')
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{top + row * len(CREDIT) + 22}" text-anchor="middle">{tick}%</text>')
    share_x = x(COLUMN_SHARE)
    parts.append(f'<line x1="{share_x:.1f}" y1="{top - 52}" x2="{share_x:.1f}" y2="{top + row * len(CREDIT) - 14}" stroke="var(--ink-2)" stroke-width="3" stroke-dasharray="10 8"/>')
    parts.append(f'<text class="value" x="{share_x + 14:.1f}" y="{top - 34}">SVs’ share of columns: 0.32%</text>')
    parts.append(f'<text class="axis" x="{(plot_left + plot_right) / 2:.1f}" y="{height - 6}" text-anchor="middle">SVs’ share of held-out predictive covariance</text>')
    for index, (method, (lo, hi)) in enumerate(CREDIT):
        cy = top + row * index + row / 2 - 14
        parts.append(f'<text class="row-label" x="{label_right}" y="{cy + 11:.1f}" text-anchor="end">{method}</text>')
        parts.append(
            f'<rect x="{x(lo) - 13:.1f}" y="{cy - 13:.1f}" width="{x(hi) - x(lo) + 26:.1f}" height="26" rx="13" fill="var(--ink)">'
            f"<title>{method}: {fmt(lo)}% to {fmt(hi)}%</title></rect>"
        )
        if x(hi) - x(lo) > 160:
            parts.append(f'<text class="value" x="{(x(lo) + x(hi)) / 2:.1f}" y="{cy - 26:.1f}" text-anchor="middle">{fmt(lo)}–{fmt(hi)}%</text>')
        else:
            parts.append(f'<text class="value" x="{max(x(hi) + 26, share_x + 16):.1f}" y="{cy + 10:.1f}">{fmt(lo)}–{fmt(hi)}%</text>')
    parts.append("</svg>")
    return "".join(parts)


TEMPLATE = (HERE / "poster_template.html").read_text()
html = (
    TEMPLATE.replace("{{IDEOGRAM}}", ideogram())
    .replace("{{SKEW}}", skew_chart())
    .replace("{{BASELINES}}", baseline_chart())
    .replace("{{CREDIT}}", credit_chart())
)
assert "{{" not in html
OUT.write_text(html)
print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


# ------------------------------------------------------------------ the print build: fonts inlined, so the PDF cannot fall back

import base64
import re
import urllib.request

KEEP = {"latin", "latin-ext", "greek"}
# Google Fonts serves woff2 only to a browser it recognizes.
BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"


def inlined_fonts() -> str:
    (HERE / "fonts").mkdir(exist_ok=True)
    sheet = HERE / "fonts" / "fonts.css"
    if not sheet.exists():
        url = re.search(r'href="(https://fonts.googleapis.com/css2[^"]+)"', html).group(1).replace("&amp;", "&")
        request = urllib.request.Request(url, headers={"User-Agent": BROWSER})
        sheet.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    css = sheet.read_text()
    blocks = re.findall(r"/\* ([a-z-]+) \*/\s*(@font-face \{.*?\})", css, flags=re.S)
    kept = []
    for subset, block in blocks:
        if subset not in KEEP:
            continue
        url = re.search(r"url\((https://[^)]+)\)", block).group(1)
        cache = HERE / "fonts" / url.rsplit("/", 1)[-1]
        if not cache.exists():
            cache.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        data = base64.b64encode(cache.read_bytes()).decode()
        kept.append(block.replace(url, f"data:font/woff2;base64,{data}"))
    return "\n".join(kept)


link = re.search(r'<link rel="stylesheet" href="https://fonts.googleapis.com[^"]+">', html).group(0)
print_html = (
    "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    + html.replace(link, "<style>" + inlined_fonts() + "</style>")
    .replace("<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">", "")
    .replace("<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>", "")
    .replace("<div class=\"desk\">", "</head><body style=\"margin:0\"><div class=\"desk\">", 1)
    + "</body></html>"
)
(HERE / "sv-pgs-poster-print.html").write_text(print_html)
print(f"wrote print build ({len(print_html) / 1024:.0f} KB)")
