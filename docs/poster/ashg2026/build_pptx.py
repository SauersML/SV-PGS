"""The ASHG 2026 science poster as a PowerPoint file, built on the lab's poster template (48 x 36 in): the accepted
abstract's title and authors, the realistic simulation study (benchmarks/power/grid.py on bench-sim's 1000 Genomes
cohort, run on MSI), and placeholders for the All of Us results.

    python3 build_pptx.py [template.pptx] [output.pptx] [simulation_summary.json] [figures directory]

The template is read from outside the repository and nothing of it is committed: its own content (another study's All
of Us results) must not enter this public repository. Its header, University of Minnesota wordmark and section bars are
kept; every text box, figure and caption is replaced, all in Arial at one size and in black outside the header and
bars. The author line and affiliations come from authors.local beside this file, untracked, so no personal name is
committed. The figures come from figures.py.

Placeholders: a figure the simulation or All of Us runs will supply is a dashed "placeholder" panel (figures.py); a
number they will supply is a bracketed bold run in the text, e.g. "[x%]". The simulation's numbers and the verbs that
depend on them come from simulation_summary.json (simulation.py) when it exists.
"""

from __future__ import annotations

import functools
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "ASHG draft.PH.pptx"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Downloads" / "ASHG SV-PGS poster.pptx"
SUMMARY = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "simulation_summary.json"
FIGURES = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / "figures"
SKILL = Path.home() / ".claude/skills/synced/cf317dd8-fe6e-4d5f-8d52-41758c24f59e_65ab35e1-5c99-4c20-a531-f6c547fa9553/pptx"
EMU = 914400
PLACEHOLDER = "000000"

# ------------------------------------------------------------------ the copy

TITLE = ["Enabling Polygenic Prediction from Structural", "Variants in the NIH All of Us Multi-Ancestry Biobank"]

INTRODUCTION = [
    [("Polygenic scores (PGSs) predict disease risk from the summed effects of many variants.", "")],
    [("Most PGS methods use only SNVs, although structural variants (SVs, ≥ 50 bp) also affect disease risk.", "")],
    [("We test whether SVs improve prediction in simulations and in the multi-ancestry NIH All of Us cohort.", "")],
]

METHODS = [
    [("SV-PGS fits all variants in one model and runs on GPUs.", "")],
    [("For All of Us, SVs discovered in 12,212 phased genome assemblies are imputed into short-read genomes.", "")],
]

SIMULATION_DESIGN = [
    [("50,000 people simulated from 1000 Genomes chr22 haplotypes in five ancestry groups, with SVs imputed from a separate panel.", "")],
    [("We vary prevalence (1–10%), h² (0.01–0.1), causal count (50–2,000), SV fraction (0–20%) and SV effect SD (1× or 3× SNVs).", "")],
    [("Rarer variants have larger effects (α = −0.38, as in UK Biobank); SV effects may also grow with length or depend on type.", "")],
]

SIMULATION_METHODS = [
    [("SV-PGS with/without SVs and LDpred2-auto (MAF ≥ 1% SNVs at HapMap3 density, or all classes) use 5,000–40,000 training samples.", "")],
]

SIMULATION_ACCURACY = [
    [("With 20% causal SVs and 40,000 training samples, adding SVs changed AUC by ", ""), ("[ΔAUC]", "ph"),
     (". With no causal SVs, AUC changed by ", ""), ("[ΔAUC]", "ph"), (".", "")],
    [("SV-PGS − LDpred2-auto ΔAUC was ", ""), ("[ ]", "ph"), (" (SNV subset) and ", ""), ("[ ]", "ph"), (" (all variant classes).", "")],
]

SIMULATION_EFFECTS = [
    [("With 20% causal SVs, estimated power at 40,000 samples was ", ""), ("[x%]", "ph"),
     (" (95% CI ", ""), ("[ ]", "ph"), ("; ", ""), ("[n]", "ph"), (" replicates).", "")],
    [("The correlation of SV length with estimated |effect| was ρ = ", ""), ("[ ]", "ph"),
     (" with a simulated length effect and ", ""), ("[ ]", "ph"), (" without one.", "")],
]

HYPERTENSION = [
    [("We fit SV-PGS to hypertension with over 200,000 SVs called in 97,000 short-read genomes.", "")],
    [("About as many SVs increased risk as decreased it (", ""), ("[x%]", "ph"), (" vs. ", ""), ("[y%]", "ph"),
     ("), and longer SVs had larger effects (", ""), ("[statistic]", "ph"), (").", "")],
]

DISEASES = [
    [("Averaged over ", ""), ("[n]", "ph"), (" diseases, adding SVs changed test AUC by ", ""), ("[ΔAUC]", "ph"),
     (" (95% CI ", ""), ("[ ]", "ph"), (").", "")],
]

DISCUSSION = [
    [("Adding SVs ", ""), ("[improved / did not improve]", "ph"), (" prediction in simulations with causal SVs and in ", ""),
     ("[n]", "ph"), (" All of Us diseases.", "")],
    [("SV-PGS is open-source software for training and applying PGSs that include SVs.", "")],
]

# ------------------------------------------------------------------ the simulation's numbers in the copy


def number(value: float, digits: int = 3) -> str:
    """A value with a typographic minus sign; one that rounds to zero prints unsigned, not as −0.000."""
    text = f"{value:.{digits}f}"
    return text.lstrip("-") if float(text) == 0.0 else text.replace("-", "−")


def auc_change(entry: dict) -> str:
    """'0.006 (95% CI 0.004–0.008)' style text for an AUC difference and its interval."""
    low, high = entry["lower"], entry["upper"]
    span = f"{number(low)}–{number(high)}" if low * high > 0 else f"{number(low)} to {number(high)}"
    return f"{number(entry['mean'])} (95% CI {span})"


MIN_REPLICATES = 3
"""A sentence is filled only from (point, training size) cells with at least this many finished replicates."""


def settled(point: dict, size: str) -> dict | None:
    """The point's cell at ``size`` if it has enough replicates for an interval, else None."""
    cell = point.get("sizes", {}).get(size)
    return cell if cell and cell["replicates"] >= MIN_REPLICATES else None


def simulation_copy(summary: dict) -> None:
    """Fill the simulation's bracketed numbers with estimates and uncertainty from completed replicates."""
    global SIMULATION_METHODS, SIMULATION_ACCURACY, SIMULATION_EFFECTS, DISCUSSION
    points = summary["points"]
    largest = str(max(summary["train_counts"]))
    largest_text = f"{int(largest):,}"
    cells = [cell for point in points.values() for cell in point.get("sizes", {}).values()]
    counts = [cell["replicates"] for cell in cells]
    # A partial real run must not leave template numbers in its simulation narrative, or imply that the earliest
    # completed cells are a finished grid. Keep the planned sample-size range separate from observed coverage.
    if counts:
        SIMULATION_ACCURACY = [[(f"Interim results: {sum(counts)} paired comparisons across {len(cells)} settings and sample sizes; "
                                 f"{min(counts)}–{max(counts)} completed replicates per cell. The grid is incomplete.", "")]]
    else:
        SIMULATION_ACCURACY = [[("No paired simulation comparisons have completed yet.", "")]]
    SIMULATION_EFFECTS = [[("Power and effect-size summaries are preliminary while simulation replicates finish.", "")]]
    DISCUSSION = [[("The simulation grid is incomplete. Predictive benefit remains under evaluation; All of Us results are pending.", "")],
                  DISCUSSION[1]]
    SIMULATION_METHODS = [[("SV-PGS with/without SVs and LDpred2-auto (MAF ≥ 1% SNVs at HapMap3 density, or all classes) use "
                           f"{min(summary['train_counts']):,}–{int(largest):,} training samples.", "")]]

    twenty, null_cell = settled(points.get("sv_20pct", {}), largest), settled(points.get("null", {}), largest)
    accuracy = []
    if twenty and null_cell:
        positive, null = twenty["delta_auc"], null_cell["delta_auc"]
        accuracy.append([
            (f"At {largest_text} samples, adding SVs changed AUC by {auc_change(positive)} with 20% causal SVs; "
             f"without causal SVs, ΔAUC = {number(null['mean'])}.", ""),
        ])
    if twenty:
        found = [(twenty[key], which) for key, which in (("delta_auc_vs_ldpred2", "SNV subset"),
                                                         ("delta_auc_vs_ldpred2_allvar", "all variant classes"))
                 if twenty.get(key, {}).get("replicates", 0) >= MIN_REPLICATES]
        if found:
            gains = " and ".join(f"{auc_change(against)} ({which})" for against, which in found)
            accuracy.append([(f"At 20% causal SVs, ΔAUC vs. LDpred2-auto was {gains}.", "")])
    if accuracy:
        SIMULATION_ACCURACY = accuracy

    effects = []
    if twenty:
        power = twenty["power"]
        if power["replicates"] >= MIN_REPLICATES:
            effects.append([(f"At 20% causal SVs and {largest_text} samples, power was {power['rate']:.0%} "
                             f"(95% CI {power['lower']:.0%}–{power['upper']:.0%}; {power['replicates']} replicates).", "")])
    length_cell = settled(points.get("length_positive", {}), largest)
    if length_cell and twenty:
        rho = length_cell["length_rho"][0]
        null_rho = twenty["length_rho"][0]
        effects.append([(f"SV length–|effect| correlation was ρ = {number(rho, 2)} with a length effect "
                         f"and {number(null_rho, 2)} without one.", "")])
    if effects:
        SIMULATION_EFFECTS = effects

    positives = [cell["delta_auc"] for cell in (settled(point, largest) for name, point in points.items() if name != "null") if cell]
    if positives:
        improved = sum(entry["lower"] > 0 for entry in positives)
        DISCUSSION = [
            [(f"At {largest_text} samples, {improved}/{len(positives)} causal-SV settings with ≥{MIN_REPLICATES} replicates "
              "had a positive gain interval (unadjusted 95% CI). Results are preliminary.", "")],
            DISCUSSION[1],
        ]


if SUMMARY.is_file():
    simulation_copy(json.loads(SUMMARY.read_text()))


REFERENCES = [
    "Byrska-Bishop et al. (2022) Cell 185:3426.",
    "Taylor et al. (2024) Nature 631:610.",
    "Privé et al. (2020) Bioinformatics 36:5424.",
    "All of Us Research Program Investigators (2019) N Engl J Med 381:668.",
]

BARS = {97: "Introduction", 91: "Methods", 93: "Simulation design", 118: "Simulation results", 119: "All of Us",
        95: "Discussion", 96: "References"}

# ------------------------------------------------------------------ DrawingML in the template's own formats

FONT = '<a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/><a:sym typeface="Arial"/>'


def run(text: str, flags: str, size: int, colour: str = "000000", bold: bool = False) -> str:
    flags_set = set(flags.split())
    if "ph" in flags_set:
        colour, bold = PLACEHOLDER, True
    baseline = ' baseline="30000"' if "sup" in flags_set else (' baseline="-25000"' if "sub" in flags_set else "")
    b = "1" if bold or "b" in flags_set else "0"
    i = "1" if "i" in flags_set else "0"
    space = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{size}" b="{b}" i="{i}" u="none" strike="noStrike" cap="none"{baseline} dirty="0">'
        f'<a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>{FONT}</a:rPr><a:t{space}>{escape(text, quote=False)}</a:t></a:r>'
    )


def bullet(runs: list[tuple[str, str]], size: int = 4900, after: int = 600) -> str:
    return (
        '<a:p><a:pPr marL="50800" marR="0" lvl="0" indent="-311150" algn="l" rtl="0"><a:lnSpc><a:spcPct val="90000"/></a:lnSpc>'
        f'<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="{after}"/></a:spcAft><a:buClr><a:srgbClr val="000000"/></a:buClr>'
        f'<a:buSzPts val="{size}"/><a:buFont typeface="Arial"/><a:buChar char="•"/></a:pPr>'
        + "".join(run(text, flags, size) for text, flags in runs)
        + "</a:p>"
    )


def heading(text: str, size: int = 4900) -> str:
    return (
        '<a:p><a:pPr marL="0" marR="0" lvl="0" indent="0" algn="l" rtl="0"><a:lnSpc><a:spcPct val="90000"/></a:lnSpc>'
        '<a:spcBef><a:spcPts val="600"/></a:spcBef><a:spcAft><a:spcPts val="600"/></a:spcAft><a:buNone/></a:pPr>'
        + run(text, "", size, bold=True)
        + "</a:p>"
    )


def line(runs: list[tuple[str, str]], size: int, colour: str, after: int = 0, spacing: int = 100000) -> str:
    return (
        f'<a:p><a:pPr marL="0" marR="0" lvl="0" indent="0" algn="l" rtl="0"><a:lnSpc><a:spcPct val="{spacing}"/></a:lnSpc>'
        f'<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="{after}"/></a:spcAft><a:buNone/></a:pPr>'
        + "".join(run(text, flags, size, colour) for text, flags in runs)
        + "</a:p>"
    )


def caption(text: str) -> str:
    # the template's caption style (italic, dark text), at the poster's one text size
    return (
        '<a:p><a:pPr marL="0" lvl="0" indent="0" algn="l" rtl="0"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
        f'<a:r><a:rPr lang="en-US" sz="4900" i="1"><a:solidFill><a:srgbClr val="000000"/></a:solidFill>{FONT}</a:rPr><a:t>{escape(text, quote=False)}</a:t></a:r></a:p>'
    )


def body(items: list) -> str:
    return "".join(heading(item[1]) if isinstance(item, tuple) and item[0] == "head" else bullet(item) for item in items)


LIBERATION = Path("/Applications/LibreOffice.app/Contents/Resources/fonts/truetype")
LINE_PITCH = 1.066
"""Line pitch per point of type at the bullets' 90% spacing, measured on the LibreOffice render."""
BULLET_INDENT = 0.35
"""Inches from a text box's inset edge to a bullet's text, measured on the LibreOffice render."""


@functools.cache
def face(bold: bool, italic: bool):
    from PIL import ImageFont

    name = "LiberationSans-" + {(False, False): "Regular", (True, False): "Bold", (False, True): "Italic", (True, True): "BoldItalic"}[bold, italic]
    return ImageFont.truetype(str(LIBERATION / f"{name}.ttf"), 100)


def wrapped_lines(runs: list[tuple[str, str]], width: float, size_pt: float, italic: bool = False) -> int:
    """Lines a paragraph of ``runs`` takes at ``width`` inches, wrapped greedily at spaces."""
    scale = size_pt / 100 / 72
    words: list[tuple[str, bool]] = []
    for text, flags in runs:
        bold = "ph" in flags.split() or "b" in flags.split()
        for index, piece in enumerate(text.split(" ")):
            if index:
                words.append((" ", bold))
            if piece:
                words.append((piece, bold))
    lines, used = 1, 0.0
    pending = 0.0
    for piece, bold in words:
        length = face(bold, italic).getlength(piece) * scale
        if piece == " ":
            pending += length
            continue
        if used and used + pending + length > width:
            lines, used = lines + 1, length
        else:
            used += pending + length
        pending = 0.0
    return lines


def insets(xml: str, shape_id: int) -> tuple[float, float, float, float]:
    start, end = shape_span(xml, shape_id)
    found = re.search(r'<a:bodyPr[^>]*lIns="(\d+)" tIns="(\d+)" rIns="(\d+)" bIns="(\d+)"', xml[start:end])
    return tuple(int(value) / EMU for value in found.groups()) if found else (0.1, 0.05, 0.1, 0.05)


def authors_and_affiliations() -> tuple[list[tuple[str, str]], list[str]]:
    """authors.local: "Name | 1,2" per author in order (an empty mark list for a group author), then affiliation lines
    that start with their number."""
    authors, affiliations = [], []
    for raw in (HERE / "authors.local").read_text().splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if "|" in text:
            name, marks = (part.strip() for part in text.split("|", 1))
            authors.append((name, marks))
        else:
            affiliations.append(text)
    return authors, affiliations


# ------------------------------------------------------------------ editing the slide

def shape_span(xml: str, shape_id: int) -> tuple[int, int]:
    for kind in ("sp", "pic"):
        match = re.search(rf'<p:{kind}>(?:(?!</p:{kind}>).)*?<p:cNvPr id="{shape_id}" .*?</p:{kind}>', xml, flags=re.S)
        if match:
            return match.span()
    raise KeyError(shape_id)


def replace_text(xml: str, shape_id: int, paragraphs: str) -> str:
    start, end = shape_span(xml, shape_id)
    shape = xml[start:end]
    body_start = shape.index("<a:lstStyle/>") + len("<a:lstStyle/>")
    body_end = shape.index("</p:txBody>")
    return xml[:start] + shape[:body_start] + paragraphs + shape[body_end:] + xml[end:]


def move(xml: str, shape_id: int, x: float, y: float, w: float, h: float) -> str:
    start, end = shape_span(xml, shape_id)
    shape = re.sub(
        r'<a:off x="-?\d+" y="-?\d+"/><a:ext cx="\d+" cy="\d+"/>',
        f'<a:off x="{round(x * EMU)}" y="{round(y * EMU)}"/><a:ext cx="{round(w * EMU)}" cy="{round(h * EMU)}"/>',
        xml[start:end], count=1,
    )
    return xml[:start] + shape + xml[end:]


def retitle(xml: str, shape_id: int, text: str) -> str:
    """Set a section bar's label, keeping its run formatting."""
    start, end = shape_span(xml, shape_id)
    shape = re.sub(r"<a:t>[^<]*</a:t>", f"<a:t>{escape(text, quote=False)}</a:t>", xml[start:end], count=1)
    return xml[:start] + shape + xml[end:]


def delete(xml: str, shape_id: int) -> str:
    start, end = shape_span(xml, shape_id)
    return xml[:start] + xml[end:]


def picture(xml: str, shape_id: int, rel: str) -> str:
    """Point picture ``shape_id`` at relationship ``rel``, uncropped (the new figure fills its frame)."""
    start, end = shape_span(xml, shape_id)
    shape = re.sub(r'r:embed="[^"]+"', f'r:embed="{rel}"', xml[start:end], count=1)
    shape = re.sub(r"<a:srcRect[^>]*/>", "<a:srcRect/>", shape, count=1)
    return xml[:start] + shape + xml[end:]


def duplicate(xml: str, shape_id: int, new_id: int, name: str) -> str:
    start, end = shape_span(xml, shape_id)
    shape = re.sub(r'<p:cNvPr id="\d+" name="[^"]*"', f'<p:cNvPr id="{new_id}" name="{name}"', xml[start:end], count=1)
    return xml[:end] + shape + xml[end:]


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[16:24]
    return int.from_bytes(data[:4], "big"), int.from_bytes(data[4:], "big")


def build() -> None:
    work = Path(tempfile.mkdtemp(prefix="svpgs_pptx_"))
    with zipfile.ZipFile(TEMPLATE) as archive:
        archive.extractall(work)
    slide_path = work / "ppt/slides/slide1.xml"
    rels_path = work / "ppt/slides/_rels/slide1.xml.rels"
    xml = slide_path.read_text()
    rels = rels_path.read_text()

    # the old study's comments go with its content
    rels = re.sub(r'<Relationship Id="[^"]+" Type="[^"]*comments" Target="[^"]+"/>', "", rels)
    shutil.rmtree(work / "ppt/comments")
    types = work / "[Content_Types].xml"
    types.write_text(re.sub(r'<Override PartName="/ppt/comments/[^"]+"[^>]*/>', "", types.read_text()))

    # the figures as media of their own
    figures = {
        "equation": "equation.png", "accuracy": "ph_sim_accuracy.png",
        "power": "ph_sim_power.png", "effects": "ph_sim_effects.png", "hypertension": "ph_aou_hypertension.png",
        "diseases": "ph_aou_diseases.png",
    }
    # the per-scenario gains exist once the grid has more than one scenario (figures.py)
    if (FIGURES / "fig_sim_architectures.png").is_file():
        figures["architectures"] = "fig_sim_architectures.png"
    ids = {}
    for index, (key, name) in enumerate(figures.items()):
        target = f"svpgs_{name}"
        shutil.copy(FIGURES / name, work / "ppt/media" / target)
        ids[key] = f"rIdSvpgs{index + 1}"
        rels = rels.replace(
            "</Relationships>",
            f'<Relationship Id="{ids[key]}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/></Relationships>',
        )

    # header: the abstract's title (two lines, as the template sets it), then its authors and affiliations
    title = "".join(
        '<a:p><a:pPr marL="0" lvl="0" indent="0" algn="l" rtl="0"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
        + f'<a:r><a:rPr lang="en-US" sz="7600" b="1" dirty="0"><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>{FONT}</a:rPr><a:t>{escape(text)}</a:t></a:r></a:p>'
        for text in TITLE
    )
    xml = replace_text(xml, 89, title)
    authors, affiliations = authors_and_affiliations()
    author_runs: list[tuple[str, str]] = []
    for position, (name, marks) in enumerate(authors):
        author_runs.append((name, ""))
        if marks:
            author_runs.append((marks.replace(" ", ""), "sup"))
        if position < len(authors) - 1:
            author_runs.append((", ", ""))
    plain = "".join(text for text, _flags in author_runs)
    # the template's three-name line (about 56 characters at 72 pt) clears the wordmark with room for some 64
    names_size = min(7200, 7200 * 64 // max(len(plain), 1) // 100 * 100)
    affiliation_lines = [
        line([(entry.split(" ", 1)[0], "sup"), (entry.split(" ", 1)[1], "")], 2200, "FFFFFF") for entry in affiliations
    ]
    xml = replace_text(xml, 90, line(author_runs, names_size, "FFFFFF", after=300) + "".join(affiliation_lines))
    xml = move(xml, 90, 1.37, 2.55, 32.5, 2.9)
    for patch in (102, 103, 104, 105, 108, 111):
        xml = delete(xml, patch)

    # the template's shapes, duplicated where the poster needs more of a kind: section bars, pictures, captions, text
    for source, new, name in (
        (93, 118, "Simulation results bar"), (93, 119, "All of Us bar"),
        (106, 120, "Simulation effects"), (106, 121, "Hypertension"), (106, 122, "Diseases"),
        (113, 123, "Simulation effects caption"), (113, 124, "Hypertension caption"), (113, 125, "Diseases caption"),
        (94, 126, "Simulation accuracy text"), (94, 127, "Simulation effects text"), (94, 128, "Hypertension text"),
        (94, 129, "Diseases text"), (106, 130, "Simulation architectures"), (113, 131, "Simulation architectures caption"),
        (94, 132, "Simulation methods text"),
    ):
        xml = duplicate(xml, source, new, name)
    for bar, text in BARS.items():
        xml = retitle(xml, bar, text)

    class Column:
        """Stacks a column's bars, text boxes, figures and captions top down, each text box as tall as its wrapped
        text, and refuses to run past the poster's bottom margin."""

        def __init__(self, x: float, width: float) -> None:
            self.x, self.width, self.y = x, width, 6.0

        def bar(self, xml: str, shape_id: int) -> str:
            xml = move(xml, shape_id, self.x, self.y, self.width, 0.9)
            self.y += 1.0
            return xml

        def text(self, xml: str, shape_id: int, items: list, paragraphs=body, indent: float = BULLET_INDENT) -> str:
            left, top, right, bottom = insets(xml, shape_id)
            room = self.width - left - right - indent
            height = top + bottom + sum(wrapped_lines(item, room, 49) * 49 * LINE_PITCH / 72 + 6 / 72 for item in items)
            xml = move(xml, shape_id, self.x, self.y, self.width, height)
            self.y += height + 0.15
            return replace_text(xml, shape_id, paragraphs(items))

        def figure(self, xml: str, shape_id: int, key: str, width: float | None = None, x: float | None = None) -> str:
            width = width or self.width - 0.25
            pixels_w, pixels_h = png_size(FIGURES / figures[key])
            height = width * pixels_h / pixels_w
            xml = move(picture(xml, shape_id, ids[key]), shape_id, self.x + 0.12 if x is None else x, self.y, width, height)
            self.y += height
            return xml

        def caption(self, xml: str, shape_id: int, text: str, width: float | None = None, x: float | None = None) -> str:
            width = width or self.width - 0.25
            left, top, right, bottom = insets(xml, shape_id)
            height = top + bottom + wrapped_lines([(text, "")], width - left - right, 49, italic=True) * 49 * 1.17 / 72
            xml = move(xml, shape_id, self.x + 0.12 if x is None else x, self.y, width, height)
            self.y += height + 0.3
            return replace_text(xml, shape_id, caption(text))

        def finish(self) -> None:
            if self.y > 35.75:
                raise SystemExit(f"column at x = {self.x} in runs to {self.y:.2f} in, past the bottom margin (35.75 in)")

    # the template's own picture and caption that no section uses
    xml = delete(delete(xml, 107), 114)

    # column 1: introduction, methods with the model, the simulation's design
    column = Column(1.01, 13.6)
    xml = column.bar(xml, 97)
    xml = column.text(xml, 98, INTRODUCTION)
    xml = column.bar(xml, 91)
    xml = column.figure(xml, 110, "equation", width=13.2)
    xml = column.text(xml, 92, METHODS)
    xml = column.bar(xml, 93)
    xml = column.text(xml, 94, SIMULATION_DESIGN)
    column.finish()

    # column 2: the simulation's results, in four figures
    column = Column(14.85, 17.2)
    xml = column.bar(xml, 118)
    xml = column.text(xml, 132, SIMULATION_METHODS)
    xml = column.figure(xml, 101, "accuracy")
    xml = column.caption(xml, 112, "Test AUC (mean and 95% t interval)")
    xml = column.text(xml, 126, SIMULATION_ACCURACY)
    half = (column.width - 0.5) / 2
    top = column.y
    xml = column.figure(xml, 106, "power", width=half)
    xml = column.caption(xml, 113, "Power", width=half)
    column.y = top
    xml = column.figure(xml, 120, "effects", width=half, x=column.x + 0.37 + half)
    xml = column.caption(xml, 123, "SV effect sizes", width=half, x=column.x + 0.37 + half)
    xml = column.text(xml, 127, SIMULATION_EFFECTS)
    if "architectures" in figures:
        xml = column.figure(xml, 130, "architectures")
        xml = column.caption(xml, 131, "AUC gain by architecture (95% t interval)")
    else:
        xml = delete(delete(xml, 130), 131)
    column.finish()

    # column 3: All of Us (hypertension, then the diseases), discussion, references
    column = Column(32.25, 15.65)
    xml = column.bar(xml, 119)
    xml = column.text(xml, 128, HYPERTENSION)
    xml = column.figure(xml, 121, "hypertension")
    xml = column.caption(xml, 124, "SV effects on hypertension")
    xml = column.text(xml, 129, DISEASES)
    xml = column.figure(xml, 122, "diseases")
    xml = column.caption(xml, 125, "Test AUC by disease")
    xml = column.bar(xml, 95)
    xml = column.text(xml, 109, DISCUSSION)
    xml = column.bar(xml, 96)
    xml = column.text(xml, 99, [[(reference, "")] for reference in REFERENCES], indent=0.0,
                      paragraphs=lambda items: "".join(line(runs, 4900, "000000", after=600, spacing=90000) for runs in items))
    column.finish()

    # the section bars all at the template's 60 pt (its References bar was set smaller)
    for bar in BARS:
        start, end = shape_span(xml, bar)
        xml = xml[:start] + re.sub(r'sz="\d+"', 'sz="6000"', xml[start:end]) + xml[end:]
    # one sans-serif face throughout, which PowerPoint and LibreOffice (as metric-identical Liberation Sans) set alike
    xml = xml.replace('typeface="Century Schoolbook"', 'typeface="Arial"').replace('typeface="Merriweather"', 'typeface="Arial"')
    # the comment anchor in the slide's extensions, and every relationship the new slide no longer references (the old
    # figures, the old equation, the old references' links), so the cleanup below drops their parts too
    xml = re.sub(r"<p188:commentRel [^>]*/>", "", xml)
    xml = re.sub(r"<p:ext uri=\"[^\"]+\">\s*</p:ext>", "", xml)
    xml = re.sub(r"<p:extLst>\s*</p:extLst>", "", xml)
    kept = []
    for relationship in re.findall(r"<Relationship [^>]*/>", rels):
        rel_id = re.search(r'Id="([^"]+)"', relationship).group(1)
        structural = "slideLayout" in relationship or "notesSlide" in relationship
        if structural or f'"{rel_id}"' in xml:
            kept.append(relationship)
    rels = re.sub(r"<Relationship [^>]*/>", "", rels).replace("</Relationships>", "".join(kept) + "</Relationships>")
    slide_path.write_text(xml)
    rels_path.write_text(rels)
    subprocess.run([sys.executable, str(SKILL / "scripts/clean.py"), str(work)], check=True, capture_output=True)
    OUTPUT.unlink(missing_ok=True)
    subprocess.run(["zip", "-Xqr", str(OUTPUT), "."], cwd=work, check=True)
    shutil.rmtree(work)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
