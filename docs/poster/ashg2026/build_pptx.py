"""The ASHG 2026 science poster as a PowerPoint file, built on the lab's poster template (48 x 36 in).

    python3 build_pptx.py [template.pptx] [output.pptx]

The template is read from outside the repository and nothing of it is committed: its own content (another study's All
of Us results) must not enter this public repository. Every text box, figure and caption of the template is replaced;
its header, University of Minnesota wordmark, section bars and author block are kept, all set in Arial. The figures come from figures.py.
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

# ------------------------------------------------------------------ the copy

TITLE = ["A Bayesian polygenic score model for", "SNVs, indels and structural variants"]

INTRODUCTION = [
    [("Structural variants (SVs, ≥ 50 bp) and tandem repeats change more sequence than SNVs, yet most polygenic scores (PGS) omit them or credit them only through tagging SNVs.", "")],
    [("SVs are enriched for heritability in 31 of 105 UK Biobank traits (median 14.8×; Nguyen et al. 2026).", "")],
    [("Imputed SV genotypes are noisy measurements (median r", ""), ("2", "sup"), (" = 0.78 for common SVs; Bai et al. 2026).", "")],
    [("We are building SV-PGS, a Bayesian PGS in which every variant enters one model, and testing where SVs carry signal in public data.", "")],
]

METHODS = [
    [("g", "i"), ("c", "i sub"), (": a smooth, nonparametric effect-size distribution, learned per variant class by empirical Bayes.", "")],
    [("u", "i"), ("j", "i sub"), (": each variant’s prior scale, from its imputation reliability (κ), allele frequency (", ""), ("h", "i"),
     (") and annotations such as SV length and distance to the TSS (θ).", "")],
    [("Public data: MAGE RNA-seq with 1kGP high-coverage SNV, indel and SV calls (730 people); leave-one-superpopulation-out prediction.", "")],
]

RESULTS_TOP = [
    [("Sparse priors beat an infinitesimal model: mr.ash exceeded GBLUP by 0.025–0.029 r", ""), ("2", "sup"), (" (z = 13–19).", "")],
    [("SVs were 0.32% of variants but 1.9–3.9% of mr.ash’s held-out prediction (GBLUP: 0.2–0.3%).", "")],
    [("Every method lost ~70% of r", ""), ("2", "sup"), (" with African ancestry held out.", "")],
]

RESULTS_BOTTOM = [
    ("head", "Exonic SV effects are directional"),
    [("Exon-overlapping deletions lowered their gene’s expression (mean z = −1.33, permutation p = 0.0005); exon-overlapping insertions and duplications raised it (mean z = +0.66, p = 0.0005).", "")],
    [("SVs in gene bodies without exon overlap showed no skew (mean z = 0.05 and 0.01).", "")],
    ("head", "Imputed SVs behave like confident draws"),
    [("In 5,000 mosaic genomes built from 1kGP haplotypes and re-imputed with Beagle 5.5, SVs at MAF 1–5% had a calibration slope of 0.80 (√r", ""),
     ("2", "sup"), (" = 0.83) and a dosage-variance ratio of 0.89 (r", ""), ("2", "sup"), (" = 0.69).", "")],
    [("Adding 1kGP SVs to mr.ash raised held-out r", ""), ("2", "sup"), (" by 0.0004–0.0015 (z = 1.2–2.1 per superpopulation).", "")],
]

DISCUSSION = [
    [("Exonic SV effects carry a biological sign, which a symmetric effect prior cannot represent. A location-scale extension of the SV-PGS prior is specified.", "")],
    [("Sparse empirical-Bayes priors give SVs far more predictive weight than infinitesimal models.", "")],
    [("Imputed SV dosages need calibrated reliability in the prior.", "")],
    [("Next: SNV/indel versus SNV + SV scores for 15 diseases in ~100,000 ", ""), ("All of Us", "i"),
     (" participants, starting with hypertension, coronary artery disease, asthma, hyperlipidemia and breast cancer.", "")],
]

REFERENCES = [
    "Bai et al. (2026) Nature Genetics. https://doi.org/10.1038/s41588-026-02612-z",
    "Browning, B. L., Zhou, Y., & Browning, S. R. (2018) A one-penny imputed genome from next-generation reference panels. American Journal of Human Genetics, 103(3), 338–348.",
    "Byrska-Bishop, M. et al. (2022) High-coverage whole-genome sequencing of the expanded 1000 Genomes Project cohort including 602 trios. Cell, 185(18), 3426–3440.",
    "Kim, Y., Wang, W., Carbonetto, P., & Stephens, M. (2024) A flexible empirical Bayes approach to multiple linear regression and connections with penalized regression. Journal of Machine Learning Research, 25.",
    "Nguyen et al. (2026) MiXeR-SV. Preprint. https://pmc.ncbi.nlm.nih.gov/articles/PMC13484855/",
    "Taylor, D. J. et al. (2024) Sources of gene expression variation in a globally diverse human cohort. Nature, 631, 610–616.",
]

CAPTIONS = {112: "Held-out r² of baseline methods", 113: "SV share of prediction", 114: "SV association with own-gene expression"}

# ------------------------------------------------------------------ DrawingML in the template's own formats

FONT = '<a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/><a:sym typeface="Arial"/>'


def run(text: str, flags: str, size: int, colour: str = "000000", bold: bool = False) -> str:
    flags_set = set(flags.split())
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
        '<a:spcBef><a:spcPts val="1200"/></a:spcBef><a:spcAft><a:spcPts val="600"/></a:spcAft><a:buNone/></a:pPr>'
        + run(text, "", size, bold=True)
        + "</a:p>"
    )


def plain(text: str, size: int, colour: str, bold: bool = False, italic: bool = False, after: int = 0, spacing: int = 100000) -> str:
    return (
        f'<a:p><a:pPr marL="0" marR="0" lvl="0" indent="0" algn="l" rtl="0"><a:lnSpc><a:spcPct val="{spacing}"/></a:lnSpc>'
        f'<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="{after}"/></a:spcAft><a:buNone/></a:pPr>'
        + run(text, "i" if italic else "", size, colour, bold)
        + "</a:p>"
    )


def caption(text: str) -> str:
    # the template's caption: 38 pt italic in the theme's body font, dark text
    return (
        '<a:p><a:pPr marL="0" lvl="0" indent="0" algn="l" rtl="0"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
        f'<a:r><a:rPr lang="en-US" sz="3800" i="1"><a:solidFill><a:schemeClr val="dk1"/></a:solidFill></a:rPr><a:t>{escape(text, quote=False)}</a:t></a:r></a:p>'
    )


def body(items: list) -> str:
    parts = []
    for item in items:
        if isinstance(item, tuple) and item[0] == "head":
            parts.append(heading(item[1]))
        else:
            parts.append(bullet(item))
    return "".join(parts)


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


def duplicate(xml: str, shape_id: int, new_id: int) -> str:
    start, end = shape_span(xml, shape_id)
    shape = xml[start:end].replace(f'<p:cNvPr id="{shape_id}" name="Google Shape;{shape_id};p1"', f'<p:cNvPr id="{new_id}" name="Results continued"', 1)
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

    # the new figures as media of their own
    figures = {"baselines": "fig_baselines.png", "credit": "fig_sv_credit.png", "skew": "fig_sign_skew.png", "equation": "equation.png"}
    ids = {}
    for index, (key, name) in enumerate(figures.items()):
        target = f"svpgs_{name}"
        shutil.copy(FIGURES / name, work / "ppt/media" / target)
        rel = f"rIdSvpgs{index + 1}"
        ids[key] = rel
        rels = rels.replace(
            "</Relationships>",
            f'<Relationship Id="{rel}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/></Relationships>',
        )
    if "Extension=\"png\"" not in types.read_text():
        types.write_text(types.read_text().replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1))

    # title (two lines, as the template sets them) and the white patches laid over the old figure
    title = "".join(
        '<a:p><a:pPr marL="0" lvl="0" indent="0" algn="l" rtl="0"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>'
        + f'<a:r><a:rPr lang="en-US" sz="9600" b="1" dirty="0"><a:solidFill><a:schemeClr val="lt1"/></a:solidFill>{FONT}</a:rPr><a:t>{escape(line)}</a:t></a:r></a:p>'
        for line in TITLE
    )
    xml = replace_text(xml, 89, title)
    for patch in (102, 103, 104, 105, 108, 111):
        xml = delete(xml, patch)

    # authors: authors.local (untracked; no personal name is committed here) holds edits to the template's author line,
    # "+Name" inserted after its first author with affiliation 1, set as that author is, and "-Name" removed with its
    # affiliation marks; past the template's three names the line is set at 60 pt with 50 pt superscripts, since 72 pt
    # then runs into the wordmark
    local = HERE / "authors.local"
    edits = [line.strip() for line in local.read_text().splitlines() if line.strip()] if local.exists() else []
    added = [line[1:].strip() for line in edits if line.startswith("+")]
    removed = [line[1:].strip() for line in edits if line.startswith("-")]
    if edits:
        start, end = shape_span(xml, 90)
        authors = xml[start:end]
        run_pattern = r"<a:r><a:rPr[^>]*>(?:(?!</a:r>).)*?<a:t[^>]*>([^<]*)</a:t></a:r>"
        for name in removed:
            words = set(name.split())
            runs = list(re.finditer(run_pattern, authors, flags=re.S))
            doomed = [match for match in runs if match.group(1).strip(" \u00a0,¹²³") in words]
            if not doomed:
                raise ValueError(f"no author {name!r} on the template's author line")
            following = next((match for match in runs if match.start() >= doomed[-1].end()), None)
            for match in reversed(doomed):
                authors = authors[: match.start()] + authors[match.end():]
            if following is not None and not following.group(1)[:1].isspace() and following.group(1)[:1] != "\u00a0":
                # the name it followed ended in a space; the next name keeps one
                text = following.group(1)
                authors = authors.replace(f">{text}</a:t>", f">\u00a0{text}</a:t>", 1)
        first = re.search(r"(<a:r><a:rPr[^>]*>(?:(?!</a:r>).)*?<a:t>)[^<]+,(</a:t></a:r>)((<a:r><a:rPr[^>]*>(?:(?!</a:r>).)*?<a:t>)¹(</a:t></a:r>))", authors, flags=re.S)
        names = "".join(first.group(1) + "\u00a0" + escape(name) + "," + first.group(2) + first.group(4) + "¹" + first.group(5) for name in added)
        authors = authors[: first.end()] + names + authors[first.end():]
        if 3 + len(added) - len(removed) > 3:
            authors = authors.replace('sz="6000"', 'sz="5000"').replace('sz="7200"', 'sz="6000"')
        xml = xml[:start] + authors + xml[end:]

    # column 1: introduction; methods with the model equation first
    xml = replace_text(xml, 98, body(INTRODUCTION))
    eq_w, eq_h = png_size(FIGURES / "equation.png")
    eq_width = 11.6
    eq_height = eq_width * eq_h / eq_w
    xml = picture(xml, 110, ids["equation"])
    xml = move(xml, 110, 1.07, 22.85, eq_width, eq_height)
    xml = replace_text(xml, 92, body(METHODS))
    xml = move(xml, 92, 1.07, 22.85 + eq_height + 0.15, 12.32, 35.6 - (22.85 + eq_height + 0.15))

    # column 2: results above and below the two figures
    xml = duplicate(xml, 94, 115)
    xml = replace_text(xml, 94, body(RESULTS_TOP))
    xml = move(xml, 94, 13.35, 7.01, 18.71, 5.3)
    xml = picture(xml, 101, ids["baselines"])
    xml = move(xml, 101, 13.61, 12.85, 10.94, 8.40)
    xml = picture(xml, 106, ids["credit"])
    xml = move(xml, 106, 24.75, 12.85, 6.96, 8.40)
    xml = replace_text(xml, 115, body(RESULTS_BOTTOM))
    xml = move(xml, 115, 13.35, 22.3, 18.71, 13.4)

    # column 3: the main figure, discussion, references
    xml = picture(xml, 107, ids["skew"])
    xml = move(xml, 107, 32.24, 6.11, 15.64, 11.76)
    for shape_id, text in CAPTIONS.items():
        xml = replace_text(xml, shape_id, caption(text))
    xml = move(xml, 112, 13.61, 21.25, 10.94, 0.92)
    xml = move(xml, 113, 24.75, 21.25, 6.96, 0.92)
    xml = replace_text(xml, 109, body(DISCUSSION))
    xml = replace_text(xml, 99, "".join(plain(reference, 2000, "000000", after=500, spacing=85000) for reference in REFERENCES))

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
    # one sans-serif face throughout, which PowerPoint and LibreOffice (as metric-identical Liberation Sans) set alike
    xml = xml.replace('typeface="Century Schoolbook"', 'typeface="Arial"').replace('typeface="Merriweather"', 'typeface="Arial"')
    slide_path.write_text(xml)
    rels_path.write_text(rels)
    # the old figure and equation images are no longer referenced
    subprocess.run([sys.executable, str(SKILL / "scripts/clean.py"), str(work)], check=True, capture_output=True)
    OUTPUT.unlink(missing_ok=True)
    subprocess.run(["zip", "-Xqr", str(OUTPUT), "."], cwd=work, check=True)
    shutil.rmtree(work)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
