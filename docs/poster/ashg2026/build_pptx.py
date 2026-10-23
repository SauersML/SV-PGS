"""The ASHG 2026 science poster as a PowerPoint file, built on the lab's poster template (48 x 36 in): the accepted
abstract's title and authors, the simulation study's gallery figures (sim_gallery.py) with the numbers they show, and
the All of Us results once they are drawn.

    uv run --with matplotlib python docs/poster/ashg2026/sim_gallery.py     # figures/gallery and its numbers.json
    uv run --no-project --with defusedxml --with lxml --with pillow --with qrcode python docs/poster/ashg2026/build_pptx.py \
        [template.pptx] [output.pptx] [figures directory]

The template defaults to ~/Downloads/ASHG draft.PH.pptx and the output to ~/Downloads/ASHG SV-PGS poster.pptx; render
the output to PDF (soffice --headless --convert-to pdf) to inspect it.

The template is read from outside the repository and nothing of it is committed: its own content (another study's All
of Us results) must not enter this public repository. Its header, University of Minnesota wordmark and section bars are
kept; every text box and figure is new, set on one grid (three columns between margins at the header's left edge) in
Arial, the gallery's face, with the gallery's Okabe-Ito colours naming the two methods. The author line and
affiliations come from authors.local beside this file, untracked, so no personal name is committed.

Every simulation number in the text is read from figures/gallery/numbers.json, which sim_gallery.py writes from the
values its figures draw. The All of Us section shows figures/aou_*.png in name order (else aou_results.py's
fig_aou_effects.png and fig_aou_diseases.png), each with the first line of its .txt namesake, if any, as its picture's
description and above that file's other lines as bullets; git ignores these files, so All of Us results stay local until
they are released. Without them the section holds a dashed placeholder. The section's bullets state the matched tests
in aou_numbers.json (its "matched" rows), which is also git-ignored.
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
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "ASHG draft.PH.pptx"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Downloads" / "ASHG SV-PGS poster.pptx"
FIGURES = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "figures"
GALLERY = FIGURES / "gallery"
SKILL = Path.home() / ".claude/skills/synced/cf317dd8-fe6e-4d5f-8d52-41758c24f59e_65ab35e1-5c99-4c20-a531-f6c547fa9553/pptx"
EMU = 914400

# ------------------------------------------------------------------ the grid, type and colours (inches, points)

MARGIN = 0.84
"""Left and right page margins: the header band's left edge."""
TOP, BOTTOM = 6.15, 35.3
"""The columns run from just under the header to the bottom margin."""
GUTTER = 0.8
LEFT, RIGHT = 11.4, 10.4
"""The outer columns' widths; the middle column, the simulation's results, takes the rest of the 48 in page. The All
of Us column is the narrowest: its figures shrink to the room it has."""
MIDDLE = 48 - 2 * MARGIN - 2 * GUTTER - LEFT - RIGHT
COLUMNS = ((MARGIN, LEFT), (MARGIN + LEFT + GUTTER, MIDDLE), (48 - MARGIN - RIGHT, RIGHT))
"""Each column's left edge and width."""
INSET = 0.3
"""Text sits this far inside its column, under the section bars' labels."""
BAR, BAR_SIZE = 1.25, 72
BODY, HEADLINE, GLOSS, REFERENCE = 56, 60, 56, 50
LINE_PITCH = 1.066
"""Line pitch per point of type at 90% line spacing, measured on the LibreOffice render."""
SHIFTED_PITCH = 0.25
"""Extra pitch per point of type on a line with a subscript or superscript."""
BULLET_INDENT = 0.42
AFTER = 14
"""Points of space after a bullet."""
GAPS = {"bar": 0.25, "block": 0.2, "section": 0.4}
"""Space under a section bar, between blocks of a section, and above the next bar (before the column is justified)."""
JUSTIFY_LIMIT = 2.0
"""The most extra space a section break takes when a column is spread to the bottom margin."""

INK, GREY, TEAL = "000000", "6E6E6E", "1482A5"
BLUE, VERMILLION = "0072B2", "D55E00"
"""sim_gallery.py's SV-PGS and LDpred2-auto colours; the bullets take the section bars' teal."""
SKY = "56B4E9"
COLOURS = {"blue": BLUE, "vermillion": VERMILLION, "grey": GREY, "sky": SKY}

# ------------------------------------------------------------------ the copy

TITLE = ["Enabling Polygenic Prediction from Structural", "Variants in the NIH All of Us Multi-Ancestry Biobank"]
REFERENCES = [
    "Byrska-Bishop 2022 Cell",
    "Taylor 2024 Nature",
    "Privé 2020 Bioinformatics",
    "Browning 2018 AJHG",
    "Wray 2010 PLoS Genet",
    "All of Us 2019 N Engl J Med",
]
"""First author, year and journal, one to a line, numbered in superscript as the text cites them."""
SUPERSCRIPT = dict(zip(range(1, 10), "¹²³⁴⁵⁶⁷⁸⁹"))
PLACEHOLDERS = [
    ("Change in test r² with SVs", "All held-out people and by genetic ancestry"),
]


def marked(text: str) -> list[tuple[str, str]]:
    """Runs of ``text``: **bold**, {{words|colour}} bold in a figure colour (a genotype source named as its bars are
    drawn), and each method's name bold in its figure colour."""
    runs = []
    for piece in re.split(r"(\*\*.+?\*\*|\{\{.+?\}\}|SV-PGS|LDpred2-auto)", text):
        if piece.startswith("**"):
            runs.append((piece[2:-2], "b"))
        elif piece.startswith("{{"):
            words, colour = piece[2:-2].split("|")
            runs.append((words, f"b {colour}"))
        elif piece in ("SV-PGS", "LDpred2-auto"):
            runs.append((piece, ""))
        elif piece:
            runs.append((piece, ""))
    return runs


def people(count: int) -> str:
    return f"{count:,}"


def copy(numbers: dict) -> dict:
    """The poster's sentences, each number from the gallery figure that shows it."""
    accuracy, wins, gain, svs = numbers["fig1"], numbers["poster_wins"], numbers["poster_sv_gain"], numbers["fig5"]
    rows = gain["rows"]  # true and imputed: [mean, lower, upper, simulations]; share: [ratio, lower, upper]
    simulations = rows["all"]["true"][3] + rows["null"]["true"][3]
    return {
        # ¹ is Byrska-Bishop 2022, the first of REFERENCES: ~9,679 SVs discovered per genome
        "introduction": ["Each genome carries ~9,700 structural variants (SVs)¹",
                         "Polygenic scores rarely include them; we test if imputed SVs help"],
        "methods": ["Empirical Bayes priors use r², frequency and SV length"],
        "code": ["Code:", "github.com/", "SauersML/SV-PGS"],
        "design": [
            f"{people(svs['people'])} genomes simulated from 1000 Genomes chr22",
            f"Baseline: liability h² = {accuracy['h2']:g}, prevalence {accuracy['prevalence']:.0%}, "
            f"{accuracy['causal']} causal variants, {accuracy['baseline_sv_share']:.0%} SVs; "
            f"{gain['shown'] - 1} other architectures vary these",
            "LDpred2-auto: common SNVs at HapMap3 density",
        ],
        "whole_note": f"Simulations on whole chr22, tested on {people(accuracy['test_people'])} people",
        "gain_note": f"Mean and 95% CI over {simulations} simulations in 3 Mb of chr22",
        "aou": ["SVs imputed into 50,000 genomes from 12,212 long-read assemblies", *aou_matched()],
        "discussion": [
            f"SV-PGS gained {wins['gain'][0]:.3f} AUC over LDpred2-auto on whole chr22",
            "The gain was as large without causal SVs and grew with h²",
        ],
    }


def aou_matched() -> list[str]:
    """The All of Us tests of SV-PGS with SVs against the same model fitted without SVs by identical code, on the
    same held-out people (aou_numbers.json's "matched" rows: aggregates, every count > 20): what each compares, its
    held-out people in each genetic ancestry group the figure shows, and its test partial r² without SVs, the scale of
    the change figures/aou_1_gain.png shows."""
    path = HERE / "aou_numbers.json"
    rows = json.loads(path.read_text()).get("matched", {}).get("rows", []) if path.is_file() else []
    lines = []
    for row in rows:
        lines.append(f"{row['trait']}: SV-PGS fit with and without SVs by identical code, tested on the same "
                     f"{people(row['held_out'])} people")
        if row.get("ancestries"):
            lines.append("Of them, " + ", ".join(f"{people(group['held_out'])} {group['group']}"
                                                  for group in row["ancestries"]))
        lines.append(f"Test {row['metric']} without SVs: {row['snv_indel_r2']:.3f}")
    return lines


# ------------------------------------------------------------------ DrawingML

FONT = '<a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/><a:sym typeface="Arial"/>'


def emu(inches: float) -> int:
    return round(inches * EMU)


def run_size(flags: str, size: float) -> float:
    """A run's size: ``size``, or the points its "ptNN" flag names."""
    return next((float(flag[2:]) for flag in flags.split() if flag.startswith("pt")), size)


def run(text: str, flags: str, size: float, colour: str = INK) -> str:
    flags_set = set(flags.split())
    size = run_size(flags, size)
    colour = next((COLOURS[flag] for flag in flags_set if flag in COLOURS), colour)
    baseline = ' baseline="30000"' if "sup" in flags_set else (' baseline="-25000"' if "sub" in flags_set else "")
    space = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{round(size * 100)}" b="{int("b" in flags_set)}" i="{int("i" in flags_set)}" '
        f'u="none" strike="noStrike" cap="none"{baseline} dirty="0"><a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>'
        f'{FONT}</a:rPr><a:t{space}>{escape(text, quote=False)}</a:t></a:r>'
    )


@dataclass
class Para:
    runs: list[tuple[str, str]]
    size: float = BODY
    bullet: bool = True
    colour: str = INK
    after: float = AFTER
    align: str = "l"

    def xml(self) -> str:
        spacing = (f'<a:lnSpc><a:spcPct val="90000"/></a:lnSpc><a:spcBef><a:spcPts val="0"/></a:spcBef>'
                   f'<a:spcAft><a:spcPts val="{round(self.after * 100)}"/></a:spcAft>')
        if self.bullet:
            properties = (f'<a:pPr marL="{emu(BULLET_INDENT)}" marR="0" lvl="0" indent="{-emu(BULLET_INDENT)}" '
                          f'algn="{self.align}" rtl="0">{spacing}<a:buClr><a:srgbClr val="{TEAL}"/></a:buClr>'
                          '<a:buSzPct val="100000"/><a:buFont typeface="Arial"/><a:buChar char="•"/></a:pPr>')
        else:
            properties = (f'<a:pPr marL="0" marR="0" lvl="0" indent="0" algn="{self.align}" rtl="0">{spacing}'
                          '<a:buNone/></a:pPr>')
        return "<a:p>" + properties + "".join(run(text, flags, self.size, self.colour) for text, flags in self.runs) + "</a:p>"

    def height(self, width: float) -> float:
        room = width - (BULLET_INDENT if self.bullet else 0.0)
        shifted = any({"sub", "sup"} & set(flags.split()) for _, flags in self.runs)
        tallest = max(run_size(flags, self.size) for _, flags in self.runs)
        return wrapped_lines(self.runs, room, self.size) * tallest * (LINE_PITCH + SHIFTED_PITCH * shifted) / 72


def bullets(lines: list[str], size: float = BODY) -> list[Para]:
    return [Para(marked(text), size) for text in lines]


LIBERATION = Path("/Applications/LibreOffice.app/Contents/Resources/fonts/truetype")


@functools.cache
def face(bold: bool, italic: bool):
    from PIL import ImageFont

    name = "LiberationSans-" + {(False, False): "Regular", (True, False): "Bold", (False, True): "Italic", (True, True): "BoldItalic"}[bold, italic]
    return ImageFont.truetype(str(LIBERATION / f"{name}.ttf"), 100)


def wrapped_lines(runs: list[tuple[str, str]], width: float, size_pt: float) -> int:
    """Lines a paragraph of ``runs`` takes at ``width`` inches, wrapped greedily at spaces."""
    words: list[tuple[str, bool, bool, float]] = []
    for text, flags in runs:
        bold, italic = "b" in flags.split(), "i" in flags.split()
        scale = run_size(flags, size_pt) / 100 / 72
        for index, piece in enumerate(text.split(" ")):
            if index:
                words.append((" ", bold, italic, scale))
            if piece:
                words.append((piece, bold, italic, scale))
    lines, used, pending = 1, 0.0, 0.0
    for piece, bold, italic, scale in words:
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


def textbox(shape_id: int, name: str, x: float, y: float, width: float, height: float, paragraphs: str) -> str:
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" lIns="{emu(INSET)}" tIns="0" rIns="{emu(INSET)}" bIns="0" anchor="t" '
        f'anchorCtr="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>{paragraphs}</p:txBody></p:sp>'
    )


def image(shape_id: int, name: str, rel: str, x: float, y: float, width: float, height: float, description: str) -> str:
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{escape(name)}" descr="{escape(description)}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rel}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr></p:pic>'
    )


def placeholder(shape_id: int, x: float, y: float, width: float, height: float, title: str, detail: str) -> str:
    """A dashed panel where an All of Us figure will go."""
    text = "".join(
        Para([(words, flags)], size, bullet=False, colour=colour, after=after, align="ctr").xml()
        for words, flags, size, colour, after in (("All of Us · result pending", "b", REFERENCE, TEAL, 18),
                                                  (title, "b", HEADLINE, INK, 12), (detail, "", GLOSS, GREY, 0))
    )
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="All of Us placeholder"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>'
        '<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 4000"/></a:avLst></a:prstGeom>'
        '<a:solidFill><a:srgbClr val="F4F6F8"/></a:solidFill><a:ln w="38100"><a:solidFill><a:srgbClr val="BDBDBD"/>'
        '</a:solidFill><a:prstDash val="dash"/></a:ln></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" lIns="{emu(0.5)}" tIns="{emu(0.3)}" rIns="{emu(0.5)}" bIns="{emu(0.3)}" '
        f'anchor="ctr" anchorCtr="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>{text}</p:txBody></p:sp>'
    )


# ------------------------------------------------------------------ the template's slide

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
        f'<a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/>',
        xml[start:end], count=1,
    )
    return xml[:start] + shape + xml[end:]


def delete(xml: str, shape_id: int) -> str:
    start, end = shape_span(xml, shape_id)
    return xml[:start] + xml[end:]


def duplicate(xml: str, shape_id: int, new_id: int, name: str) -> str:
    start, end = shape_span(xml, shape_id)
    shape = re.sub(r'<p:cNvPr id="\d+" name="[^"]*"', f'<p:cNvPr id="{new_id}" name="{name}"', xml[start:end], count=1)
    return xml[:end] + shape + xml[end:]


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[16:24]
    return int.from_bytes(data[:4], "big"), int.from_bytes(data[4:], "big")


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


def header(xml: str) -> str:
    """The abstract's title (two lines, as the template sets it), then its authors and affiliations."""
    title = "".join(Para([(text, "b")], 92, bullet=False, colour="FFFFFF", after=0).xml() for text in TITLE)
    xml = replace_text(xml, 89, title.replace('<a:spcPct val="90000"/>', '<a:spcPct val="100000"/>'))
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
    names_size = min(72, 72 * 64 // max(len(plain), 1))
    lines = [Para(author_runs, names_size, bullet=False, colour="FFFFFF", after=3)]
    lines += [Para([(entry.split(" ", 1)[0], "sup"), (entry.split(" ", 1)[1], "")], 34, bullet=False, colour="FFFFFF",
                   after=0) for entry in affiliations]
    xml = replace_text(xml, 90, "".join(line.xml() for line in lines))
    # 3.1 in down, so the title's descenders clear the author line
    return move(xml, 90, 1.37, 3.1, 32.5, 2.45)


# ------------------------------------------------------------------ laying out the columns

@dataclass
class Block:
    """One thing in a column: a section bar, a text box, a figure, a dashed placeholder or a row of two columns.
    ``flex`` blocks share whatever height the column leaves."""

    kind: str
    content: object = None
    flex: bool = False
    gap: str = "block"
    children: list = field(default_factory=list)
    legend: Path | None = None
    """A figure's legend, whose first sentence is its picture's description."""


def gap(previous: Block, block: Block) -> float:
    return GAPS["bar"] if previous.kind == "bar" else GAPS[block.gap]


def bar(shape_id: int, text: str) -> Block:
    return Block("bar", (shape_id, text), gap="section")


def text(paragraphs: list[Para]) -> Block:
    return Block("text", paragraphs)


QR = 2.6
"""Inches: the side of the QR code to the repository."""


def code(lines: list[str]) -> Block:
    """A QR code to the repository with its address beside it."""
    return Block("qr", [Para([(line, "b" if index else "")], BODY, bullet=False, after=0) for index, line in enumerate(lines)])


def row(*children: list, shares: tuple = ()) -> Block:
    """Stacks side by side, a gutter apart, each taking its share of the width (equal shares by default)."""
    return Block("row", shares or (1 / len(children),) * len(children), children=list(children))


def parts(block: Block, width: float) -> list[float]:
    """The widths of a row's stacks."""
    room = width - GUTTER * (len(block.children) - 1)
    return [room * share for share in block.content]


def figure(path: Path, flex: bool = False, legend: Path | None = None) -> Block:
    return Block("figure", path, flex=flex, legend=legend or path.with_suffix(".txt"))


class Poster:
    def __init__(self, xml: str, rels: str, media: Path) -> None:
        self.xml, self.rels, self.media = xml, rels, media
        self.shapes: list[str] = []
        self.next_id = 200
        self.images: dict[Path, str] = {}
        self.qr = Path(tempfile.mkdtemp(prefix="svpgs_qr_")) / "qr.png"
        import qrcode

        qrcode.make("https://github.com/SauersML/SV-PGS", border=1, box_size=40).save(self.qr)

    def fresh(self) -> int:
        self.next_id += 1
        return self.next_id

    def image(self, path: Path) -> str:
        """The relationship id of ``path`` copied into the deck's media, without an alpha channel when it is opaque (so
        a PDF export draws no soft mask over the white page)."""
        if path not in self.images:
            from PIL import Image

            rel = f"rIdSvpgs{len(self.images) + 1}"
            with Image.open(path) as picture:
                if picture.mode == "RGBA" and picture.getchannel("A").getextrema() == (255, 255):
                    picture.convert("RGB").save(self.media / f"svpgs_{path.name}")
                else:
                    shutil.copy(path, self.media / f"svpgs_{path.name}")
            self.rels = self.rels.replace(
                "</Relationships>",
                f'<Relationship Id="{rel}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="../media/svpgs_{path.name}"/></Relationships>')
            self.images[path] = rel
        return self.images[path]

    # heights

    def natural(self, block: Block, width: float) -> float:
        if block.kind == "bar":
            return BAR
        if block.kind == "text":
            paragraphs = block.content
            return (sum(para.height(width - 2 * INSET) for para in paragraphs)
                    + sum(para.after for para in paragraphs[:-1]) / 72)
        if block.kind == "figure":
            pixels_w, pixels_h = png_size(block.content)
            return 0.0 if block.flex else width * pixels_h / pixels_w
        if block.kind == "placeholder":
            return 0.0
        if block.kind == "qr":
            return QR
        if block.kind == "row":
            return max(self.stack_height(child, part) for child, part in zip(block.children, parts(block, width)))
        raise ValueError(block.kind)

    def most(self, block: Block, width: float) -> float:
        """The tallest a flex block can use: a figure as tall as the column's width lets it be, a placeholder any."""
        if block.kind == "figure":
            pixels_w, pixels_h = png_size(block.content)
            return width * pixels_h / pixels_w
        return math.inf

    def stack_height(self, blocks: list[Block], width: float) -> float:
        return (sum(self.natural(block, width) for block in blocks)
                + sum(gap(previous, block) for previous, block in zip(blocks, blocks[1:])))

    # placing

    def place(self, block: Block, x: float, y: float, width: float, height: float) -> None:
        if block.kind == "bar":
            shape_id, label = block.content
            self.xml = move(self.xml, shape_id, x, y, width, BAR)
            start, end = shape_span(self.xml, shape_id)
            shape = re.sub(r"<a:t>[^<]*</a:t>", f"<a:t>{escape(label, quote=False)}</a:t>", self.xml[start:end], count=1)
            shape = re.sub(r'sz="\d+"', f'sz="{BAR_SIZE * 100}"', shape)
            shape = re.sub(r'(<a:rPr lang="en-US" sz="\d+") b="0"', r'\1 b="1"', shape)
            self.xml = self.xml[:start] + shape + self.xml[end:]
        elif block.kind == "text":
            paragraphs = "".join(para.xml() for para in block.content)
            self.shapes.append(textbox(self.fresh(), "Text", x, y, width, height, paragraphs))
        elif block.kind == "figure":
            pixels_w, pixels_h = png_size(block.content)
            shown = min(width, height * pixels_w / pixels_h)
            description = block.legend.read_text().split(". ")[0] if block.legend.is_file() else block.content.stem
            self.shapes.append(image(self.fresh(), block.content.stem, self.image(block.content), x, y, shown,
                                     shown * pixels_h / pixels_w, description))
        elif block.kind == "placeholder":
            self.shapes.append(placeholder(self.fresh(), x, y, width, height, *block.content))
        elif block.kind == "qr":
            self.shapes.append(image(self.fresh(), "Repository QR code", self.image(self.qr), x + INSET, y, QR, QR,
                                     "QR code to github.com/SauersML/SV-PGS"))
            words = sum(para.height(width - QR - 3 * INSET) for para in block.content)
            paragraphs = "".join(para.xml() for para in block.content)
            self.shapes.append(textbox(self.fresh(), "Repository", x + QR + INSET, y + (QR - words) / 2,
                                       width - QR - INSET, words, paragraphs))
        elif block.kind == "row":
            for child, part in zip(block.children, parts(block, width)):
                self.stack(child, x, y, part, None)
                x += part + GUTTER

    def stack(self, blocks: list[Block], x: float, top: float, width: float, bottom: float | None) -> float:
        """Place ``blocks`` top down; with a ``bottom``, flex blocks share the room left (a figure no taller than the
        column's width allows), then the section breaks take what remains (up to JUSTIFY_LIMIT each), so the column
        ends on the bottom margin."""
        room = (bottom - top - self.stack_height(blocks, width)) if bottom is not None else 0.0
        if room < 0:
            raise SystemExit(f"column at x = {x:.2f} in runs {-room:.2f} in past the bottom margin")
        flexible = sorted((block for block in blocks if block.flex), key=lambda block: self.most(block, width))
        heights = {}
        for index, block in enumerate(flexible):
            heights[id(block)] = min(self.most(block, width), room / (len(flexible) - index))
            room -= heights[id(block)]
        breaks = sum(block.gap == "section" for block in blocks[1:])
        spread = min(room / breaks, JUSTIFY_LIMIT) if breaks else 0.0
        y = top
        for index, block in enumerate(blocks):
            if index:
                y += gap(blocks[index - 1], block) + (spread if block.gap == "section" else 0.0)
            height = heights[id(block)] if block.flex else self.natural(block, width)
            self.place(block, x, y, width, height)
            y += height
        return y


def aou_blocks() -> list[Block]:
    """The All of Us figures drawn locally (figures/aou_*.png), each with its .txt's title and bullets, or the
    placeholders."""
    drawn = sorted(FIGURES.glob("aou_*.png")) or [path for name in ("fig_aou_effects.png", "fig_aou_diseases.png")
                                                   if (path := FIGURES / name).is_file()]
    if not drawn:
        return [Block("placeholder", entry, flex=True) for entry in PLACEHOLDERS]
    blocks = []
    for path in drawn:
        notes = path.with_suffix(".txt")
        lines = [line.strip() for line in notes.read_text().splitlines() if line.strip()] if notes.is_file() else []
        # the first line describes the picture; any others are bullets under it
        blocks.append(figure(path, flex=True))
        if lines[1:]:
            blocks.append(text(bullets(lines[1:])))
    return blocks


def build() -> None:
    numbers_path = GALLERY / "numbers.json"
    if not numbers_path.is_file():
        raise SystemExit(f"{numbers_path} is missing: run sim_gallery.py first")
    words = copy(json.loads(numbers_path.read_text()))

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

    # a white page, so the figures' white grounds meet it without an edge
    xml = re.sub(r"<p:bg>.*?</p:bg>", '<p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
                 "<a:effectLst/></p:bgPr></p:bg>", xml, count=1, flags=re.S)
    xml = header(xml)
    # every text box, figure and caption of the template's own study; its header, wordmark and bars stay
    for shape_id in (92, 94, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114):
        xml = delete(xml, shape_id)
    xml = duplicate(duplicate(xml, 93, 118, "Simulation results bar"), 93, 119, "All of Us bar")

    gallery = {name: GALLERY / f"{name}.png" for name in (
        "poster_wins", "poster_sv_gain", "poster_accuracy_by_training_size", "poster_imputation_accuracy",
        "poster_model")}
    columns = [
        [
            bar(97, "Introduction"), text(bullets(words["introduction"])),
            bar(91, "Methods"), figure(gallery["poster_model"]), text(bullets(words["methods"])),
            bar(93, "Simulation design"), text(bullets(words["design"])),
            figure(gallery["poster_imputation_accuracy"], flex=True, legend=GALLERY / "fig5_simulated_svs.txt"),
        ],
        [
            bar(118, "Simulation results"),
            Block("text", [Para([(words["whole_note"], "")], bullet=False, after=0)]),
            row([figure(gallery["poster_accuracy_by_training_size"], legend=GALLERY / "fig1_accuracy_by_training_size.txt")],
                [figure(gallery["poster_wins"], legend=GALLERY / "fig2_gain_over_ldpred2.txt")]),
            Block("text", [Para([(words["gain_note"], "")], bullet=False, after=0)], gap="section"),
            figure(gallery["poster_sv_gain"]),
            bar(95, "Discussion"), text(bullets(words["discussion"])),
        ],
        [
            bar(119, "All of Us"), text(bullets(words["aou"])), *aou_blocks(),
            bar(96, "References"),
            text([Para([(f"{SUPERSCRIPT[index]}{reference}", "")], REFERENCE, bullet=False, after=0)
                  for index, reference in enumerate(REFERENCES, start=1)]),
            code(words["code"]),
        ],
    ]
    poster = Poster(xml, rels, work / "ppt/media")
    for (x, width), blocks in zip(COLUMNS, columns):
        poster.stack(blocks, x, TOP, width, BOTTOM)
    xml = poster.xml.replace("</p:spTree>", "".join(poster.shapes) + "</p:spTree>")
    rels = poster.rels

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

