"""The ASHG 2026 science poster as a PowerPoint file, built on the lab's poster template (48 x 36 in), following the
accepted abstract: its title, author list and framing, with marked placeholders where the All of Us results will go.

    python3 build_pptx.py [template.pptx] [output.pptx]

The template is read from outside the repository and nothing of it is committed: its own content (another study's All
of Us results) must not enter this public repository. Its header, University of Minnesota wordmark and section bars are
kept; every text box, figure and caption is replaced, all in Arial. The author line and affiliations come from
authors.local beside this file, untracked, so no personal name is committed. The figures come from figures.py.

Placeholders: a figure the All of Us runs will supply is a dashed "placeholder" panel (figures.py); a number they will
supply is a bracketed red run in the text, e.g. "[x]%".
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "ASHG draft.PH.pptx"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Downloads" / "ASHG SV-PGS poster.pptx"
SKILL = Path.home() / ".claude/skills/synced/cf317dd8-fe6e-4d5f-8d52-41758c24f59e_65ab35e1-5c99-4c20-a531-f6c547fa9553/pptx"
EMU = 914400
PLACEHOLDER = "C0392B"

# ------------------------------------------------------------------ the copy, from the abstract

TITLE = ["Enabling Polygenic Prediction from Structural", "Variants in the NIH All of Us Multi-Ancestry Biobank"]

INTRODUCTION = [
    [("Polygenic scores (PGSs) aggregate variant effects to predict genetic risk for common diseases.", "")],
    [("PGSs transfer poorly across populations: effect sizes come mostly from European-ancestry participants.", "")],
    [("PGS methods are built for SNVs and overlook structural variants (SVs, ≥ 50 bp), which influence disease risk.", "")],
    [("We hypothesize that SVs collectively improve polygenic prediction, and quantify their contribution in the multi-ancestry NIH ", ""),
     ("All of Us", "i"), (" cohort.", "")],
]

METHODS = [
    [("One model for SNVs, indels and SVs, with variant-class-specific priors: effect sizes depend on SV type (e.g., insertion vs. deletion) and length.", "")],
    [("Fully GPU-accelerated training fits genome-wide PGSs at biobank scale and estimates per-variant effect sizes.", "")],
    [("SVs from a callset discovered in 12,212 phased genome assemblies, imputed into ", ""), ("All of Us", "i"), (" short-read genomes.", "")],
    [("SNV-only vs. SNV + SV models on the same people and folds: 15 diseases, 100,000 multi-ancestry participants.", "")],
]

RESULTS_TOP = [
    ("head", "Hypertension in All of Us"),
    [("Over 200,000 SVs from 97,000 short-read genomes.", "")],
    [("Risk-increasing and risk-decreasing SVs were balanced (", ""), ("[x%]", "ph"), (" vs. ", ""), ("[y%]", "ph"), (").", "")],
    [("Longer SVs had larger effect sizes (", ""), ("[statistic]", "ph"), (").", "")],
]

RESULTS_BOTTOM = [
    ("head", "15 diseases: SNV-only vs. SNV + SV"),
    [("Adding SVs changed held-out AUC by ", ""), ("[ΔAUC]", "ph"), (" (95% CI ", ""), ("[ ]", "ph"), (") averaged over diseases, and log loss by ", ""),
     ("[Δ]", "ph"), (".", "")],
]

DISCUSSION = [
    [("SVs enter one Bayesian PGS with SNVs, under priors that learn how SV type and length shape effect sizes.", "")],
    [("SV effects depend on type: in public data (MAGE, 1kGP), exonic deletions lowered and exonic insertions/duplications raised gene expression (p = 0.0005).", "")],
    [("Comparing SNV-only and SNV + SV models across 15 diseases quantifies the added value of structural variation (", ""), ("[result]", "ph"), (").", "")],
    [("SV-PGS: an open-source tool for SV-inclusive PGS training and inference.", "")],
]

REFERENCES = [
    "All of Us Research Program Investigators (2019) The “All of Us” Research Program. New England Journal of Medicine, 381, 668–676.",
    "Byrska-Bishop, M. et al. (2022) High-coverage whole-genome sequencing of the expanded 1000 Genomes Project cohort including 602 trios. Cell, 185(18), 3426–3440.",
    "Taylor, D. J. et al. (2024) Sources of gene expression variation in a globally diverse human cohort. Nature, 631, 610–616.",
]

CAPTIONS = {
    112: "SV effects on hypertension risk",
    113: "Effect size by SV length",
    114: "Public data: SV effects depend on type",
    117: "15 diseases: SNV-only vs. SNV + SV",
}

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


def heading(text: str, size: int = 5400) -> str:
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
    # the template's caption: 38 pt italic, dark text
    return (
        '<a:p><a:pPr marL="0" lvl="0" indent="0" algn="l" rtl="0"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
        f'<a:r><a:rPr lang="en-US" sz="3800" i="1"><a:solidFill><a:schemeClr val="dk1"/></a:solidFill>{FONT}</a:rPr><a:t>{escape(text, quote=False)}</a:t></a:r></a:p>'
    )


def body(items: list) -> str:
    return "".join(heading(item[1]) if isinstance(item, tuple) and item[0] == "head" else bullet(item) for item in items)


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
        "effects": "ph_hypertension_effects.png", "length": "ph_effect_by_length.png", "skew": "fig_sign_skew.png",
        "diseases": "ph_fifteen_diseases.png", "equation": "equation.png",
    }
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

    # column 1: introduction; methods with the model equation first
    xml = replace_text(xml, 98, body(INTRODUCTION))
    xml = move(xml, 91, 1.01, 18.0, 12.1, 0.99)
    eq_w, eq_h = png_size(FIGURES / "equation.png")
    eq_width = 11.6
    eq_height = eq_width * eq_h / eq_w
    xml = picture(xml, 110, ids["equation"])
    xml = move(xml, 110, 1.07, 19.25, eq_width, eq_height)
    xml = replace_text(xml, 92, body(METHODS))
    xml = move(xml, 92, 1.07, 19.25 + eq_height + 0.15, 12.32, 35.6 - (19.25 + eq_height + 0.15))

    # column 2: hypertension (two placeholder figures), then the 15-disease comparison (one placeholder figure)
    xml = replace_text(xml, 94, body(RESULTS_TOP))
    xml = move(xml, 94, 13.35, 7.01, 18.71, 4.6)
    xml = picture(xml, 101, ids["effects"])
    xml = move(xml, 101, 13.61, 11.75, 10.94, 8.40)
    xml = picture(xml, 106, ids["length"])
    xml = move(xml, 106, 24.75, 11.75, 6.96, 8.40)
    xml = move(xml, 112, 13.61, 20.15, 10.94, 0.92)
    xml = move(xml, 113, 24.75, 20.15, 6.96, 0.92)
    xml = duplicate(xml, 94, 115, "Results continued")
    xml = replace_text(xml, 115, body(RESULTS_BOTTOM))
    xml = move(xml, 115, 13.35, 21.3, 18.71, 3.0)
    xml = duplicate(xml, 106, 116, "15 diseases placeholder")
    xml = picture(xml, 116, ids["diseases"])
    xml = move(xml, 116, 13.61, 24.0, 18.20, 9.40)
    xml = duplicate(xml, 113, 117, "15 diseases caption")
    xml = move(xml, 117, 13.61, 33.4, 18.20, 0.92)

    # column 3: the public-data figure, discussion, references
    xml = picture(xml, 107, ids["skew"])
    xml = move(xml, 107, 32.24, 6.11, 15.64, 11.76)
    for shape_id, text in CAPTIONS.items():
        xml = replace_text(xml, shape_id, caption(text))
    xml = replace_text(xml, 109, body(DISCUSSION))
    xml = replace_text(xml, 99, "".join(line([(reference, "")], 2000, "000000", after=500, spacing=85000) for reference in REFERENCES))

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
