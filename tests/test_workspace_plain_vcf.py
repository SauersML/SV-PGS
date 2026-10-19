"""The in-workspace pipeline on plain imputed VCFs: one imputation tool's standard output per chromosome per batch.

The synthetic inputs are the dry run's (tests/test_workspace_pipeline.py), with every popped batch rewritten as a
plain tool would write the same dosages, and no strata sidecar or bubble.split: Beagle 5 (GT:DS, DR2, IDs '.', no INFO/CM, a genetic map), GLIMPSE2 (GT:DS:GP, INFO,
INFO/CM, SV records carrying SVTYPE/SVLEN/END) and Minimac4 (R2, which can exceed 1).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from sv_pgs.dosage_store import VARIANT_CLASS_LEGEND, DosageStore, encode_dosage_milli
from sv_pgs.store_converter import plain_batch_fields
from sv_pgs.workspace_pipeline import ARMS, STEP_NAMES, WorkspaceConfig, read_plain_sites, run_pipeline
from tests.test_workspace_pipeline import (
    BUDGET,
    CHROMOSOMES,
    HALVES,
    RECORDS,
    _bindings,
    _Recorder,
    _reported_info,
    _research_of,
    _sequencing_names,
    _Synthetic,
    workspace,
)

# The synthetic genetic map: (bp, cM) rows, with every record's genetic position it gives. The records at 1000 and 6000
# lie beyond its ends, at its first segment's rate (0.0005 cM/bp) and its last's (0.002 cM/bp).
GENETIC_MAP = ((1500, 1.0), (3500, 2.0), (5000, 5.0))
MAPPED_CM = (0.75, 1.25, 1.25, 1.75, 1.75, 1.75, 3.0, 5.0, 7.0)
FORMAT_LINES = {
    "GT": '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    "DS": '##FORMAT=<ID=DS,Number=A,Type=Float,Description="Estimated ALT dose">',
    "GP": '##FORMAT=<ID=GP,Number=G,Type=Float,Description="Genotype probabilities">',
}


@dataclasses.dataclass(frozen=True)
class _Tool:
    """One imputation tool's plain output of the synthetic dosages: its INFO declarations, whether it writes GP, the
    decimals it writes DS and GP to, and each record's INFO, ID column and alleles (on a chromosome, in a batch)."""

    info_lines: tuple[str, ...]
    probabilities: bool
    decimals: int
    info: Callable[[str, str, int], str]
    identifier: Callable[[str, int], str]
    alleles: Callable[[int], tuple[str, str]] = lambda record: (RECORDS[record][1], RECORDS[record][2])


def _beagle_info(chromosome: str, batch: str, record: int) -> str:
    return f"AF=0.3;DR2={_reported_info(record):.2f};IMP"


BEAGLE = _Tool(
    info_lines=(
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Estimated ALT Allele Frequencies">',
        '##INFO=<ID=DR2,Number=A,Type=Float,Description="Dosage R-Squared">',
        '##INFO=<ID=IMP,Number=0,Type=Flag,Description="Imputed marker">',
    ),
    probabilities=False,
    decimals=2,
    info=_beagle_info,
    identifier=lambda chromosome, record: ".",
)


def _dosage_text(dosage: float, decimals: int, probabilities: bool) -> str:
    """A sample's DS as written to ``decimals``, then its GP from that DS (on the same grid) where the tool writes GP."""
    written = float(f"{dosage:.{decimals}f}")
    second = max(written - 1.0, 0.0)
    first = written - 2.0 * second
    gp = f":{1.0 - first - second:.{decimals}f},{first:.{decimals}f},{second:.{decimals}f}" if probabilities else ""
    return f"0|0:{written:.{decimals}f}{gp}"


def _write_plain_batches(synthetic: _Synthetic, tools: dict[str, _Tool]) -> None:
    """Every imputed batch of ``synthetic`` rewritten as its half's tool writes the same dosages."""
    for half_index, (label, batches) in enumerate(HALVES):
        tool = tools[label]
        formats = ("GT", "DS", "GP") if tool.probabilities else ("GT", "DS")
        for (batch, _width), samples in zip(batches, _sequencing_names()[half_index]):
            for chromosome in CHROMOSOMES:
                lines = [
                    "##fileformat=VCFv4.2", *tool.info_lines, *(FORMAT_LINES[name] for name in formats),
                    *(f"##contig=<ID={name},length=50000000>" for name in CHROMOSOMES),
                    "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]),
                ]
                for record, (position, *_rest) in enumerate(RECORDS):
                    ref, alt = tool.alleles(record)
                    fields = [
                        _dosage_text(float(synthetic.dosages[(chromosome, _research_of(sample))][record]), tool.decimals, tool.probabilities)
                        for sample in samples
                    ]
                    lines.append("\t".join([
                        chromosome, str(position), tool.identifier(chromosome, record), ref, alt, ".", "PASS",
                        tool.info(chromosome, f"{label}/{batch}", record), ":".join(formats), *fields,
                    ]))
                (synthetic.root / label / f"{chromosome}.{batch}.vcf").write_text("\n".join(lines) + "\n")


def _write_genetic_maps(root: Path) -> str:
    """The synthetic PLINK maps, chr21's naming its chromosome 'chr21' and chr22's '22', as the Beagle zip's two folders do."""
    (root / "maps").mkdir()
    for chromosome in CHROMOSOMES:
        named = chromosome if chromosome == "chr21" else chromosome[len("chr"):]
        (root / "maps" / f"plink.{chromosome}.GRCh38.map").write_text("".join(f"{named} . {cm} {bp}\n" for bp, cm in GENETIC_MAP))
    return str(root / "maps" / "plink.{chromosome}.GRCh38.map")


def _plain(synthetic: _Synthetic, tools: dict[str, _Tool], **overrides) -> _Synthetic:
    """``synthetic`` with its batches rewritten by ``tools`` and its config set to plain input (strata and bubble.split null)."""
    _write_plain_batches(synthetic, tools)
    raw = {**synthetic.raw, "strata_directory": None, "bubble_split": None, "genetic_map": _write_genetic_maps(synthetic.root), **overrides}
    return dataclasses.replace(synthetic, raw=raw)


def _written(synthetic: _Synthetic, chromosome: str, label: str, decimals: int) -> np.ndarray:
    """A half's dosages in thousandths [records, samples] as the plain batches write them."""
    names = [name for batch in _sequencing_names()[[half for half, _batches in HALVES].index(label)] for name in batch]
    return np.array(
        [[round(float(f"{synthetic.dosages[(chromosome, _research_of(name))][record]:.{decimals}f}") * 1000) for name in names] for record in range(len(RECORDS))],
        dtype=np.uint16,
    )


def test_a_beagle_style_plain_run_builds_every_step(workspace) -> None:
    """GT:DS with DR2, IDs '.', no INFO/CM and no GP: every step runs, the genetic positions come from the map, and the
    store has no strata annotation, removes no background and says G4 was not checked."""
    synthetic, client = workspace
    plain = _plain(synthetic, {"A": BEAGLE, "B": BEAGLE})
    recorder = _Recorder()
    config = WorkspaceConfig.from_mapping(plain.raw)
    summaries = run_pipeline(config, _bindings(recorder, client), BUDGET)
    run = config.run_directory
    assert tuple(summaries) == STEP_NAMES and (run / "export" / "accuracy.tsv").is_file()

    store_summary = summaries["store"]
    assert store_summary["input"] == "plain" and store_summary["background_zeroed"] == 0 and store_summary["no_calls"] == 1
    assert store_summary["reported_r2_fields"] == ["DR2"] and store_summary["reported_r2_above_one"] == 0
    assert all(source.startswith("genetic_map plink.") and "INFO/CM is on 0 of 9" in source for source in store_summary["genetic_position"].values())
    gates = store_summary["gates"]
    assert gates["G4_format"].startswith("NOT_CHECKED on 6 of 6 imputed batch files") and gates["reported_r2"] == "PASS"
    assert gates["strata_sites_md5"].startswith("NOT_APPLICABLE") and gates["strata_ids_md5"].startswith("NOT_APPLICABLE")
    with DosageStore.open(run / "store" / "store") as store:
        assert store.manifest_attributes["gates"] == gates
        table = store.variant_table
        classes = [VARIANT_CLASS_LEGEND[code] for code in table.variant_class[: len(RECORDS)]]
        assert classes == ["snv", "deletion", "insertion", "deletion", "insertion", "deletion", "str_vntr_repeat", "snv", "insertion"]
        # Same-POS sets and the TR locus still group; there is no bubble.
        assert table.group_first[: len(RECORDS)].tolist() == [0, 1, 1, 3, 3, 3, 6, 7, 8]
        assert set(table.annotations) == {"tr_locus", "sv_length", "in_gene", "in_exon", "log_tss_distance"}
        assert table.annotations["in_exon"][: len(RECORDS)].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0]
        np.testing.assert_allclose(table.genetic_position_cm, MAPPED_CM * len(CHROMOSOMES))
        assert table.variant_ids(np.arange(len(RECORDS))) == [f"chr21:{position}:{ref}:{alt}" for position, ref, alt, *_rest in RECORDS]
        # The stored codes are the 2-decimal DS as written: nothing removed.
        half_a = _written(plain, "chr21", "A", BEAGLE.decimals)
        np.testing.assert_array_equal(store.read_codes(0, len(RECORDS), np.arange(half_a.shape[1])), encode_dosage_milli(half_a))
    # DR2 is the reported r^2 (htslib holds INFO floats as float32).
    np.testing.assert_allclose(
        np.load(run / "store" / "reported" / "chr22.npy"), [round(_reported_info(record), 2) for record in range(len(RECORDS))], rtol=np.finfo(np.float32).eps
    )
    # One fit request per arm, holding the run's one disease's and one trait's models (``fit_request``).
    assert summaries["measurement"]["recalibrated"] and len(recorder.fit_calls) == len(ARMS)
    assert all(len({name.split("/")[0] for name in request.model_names}) == 2 for request in recorder.fit_calls)


GLIMPSE2 = _Tool(
    info_lines=(
        '##INFO=<ID=RAF,Number=A,Type=Float,Description="ALT allele frequency in the reference panel">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="ALT allele frequency computed from DS/GP field across target samples">',
        '##INFO=<ID=INFO,Number=A,Type=Float,Description="IMPUTE info quality score for diploid samples">',
        '##INFO=<ID=CM,Number=1,Type=Float,Description="Genetic position">',
        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">',
        '##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="SV length">',
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">',
    ),
    probabilities=True,
    decimals=3,
    info=lambda chromosome, batch, record: ";".join([
        f"RAF=0.3;AF=0.3;INFO={_reported_info(record):.3f};CM={0.002 * RECORDS[record][0]:.4f}",
        *(("SVTYPE=DEL", "SVLEN=-600", "END=1600") if record == 0 else ()),
        *(("SVTYPE=INS", f"SVLEN={RECORDS[record][5]}") if RECORDS[record][5] is not None else ()),
    ]),
    identifier=lambda chromosome, record: f"rs{record}" if record % 2 == 0 else ".",
    # The first record is a symbolic deletion of the 600 bases after its padding base, into the synthetic gene.
    alleles=lambda record: ("A", "<DEL>") if record == 0 else (RECORDS[record][1], RECORDS[record][2]),
)


def test_a_glimpse2_style_plain_store_reads_gp_cm_and_sv_fields(workspace) -> None:
    """GT:DS:GP with INFO and INFO/CM on every record, some IDs set, SV records carrying SVTYPE/SVLEN/END: G4 runs, the
    genetic positions are INFO/CM, a symbolic deletion is typed and spanned from its INFO, and IDs name their records."""
    synthetic, client = workspace
    plain = _plain(synthetic, {"A": GLIMPSE2, "B": GLIMPSE2}, truth_calls=None, imputation_reference_ids=None, genetic_map=None)
    # Two decode processes: the plain site list reaches each through its initializer.
    two_threads = dataclasses.replace(BUDGET, cpu_threads=2)
    summaries = run_pipeline(WorkspaceConfig.from_mapping(plain.raw), _bindings(_Recorder(), client), two_threads, through="store")
    store_summary = summaries["store"]
    assert store_summary["gates"]["G4_format"] == "PASS" and store_summary["reported_r2_fields"] == ["INFO"]
    assert store_summary["genetic_position"] == {chromosome: "INFO/CM" for chromosome in CHROMOSOMES} and store_summary["background_zeroed"] == 0
    with DosageStore.open(Path(plain.raw["run_directory"]) / "store" / "store") as store:
        table = store.variant_table
        classes = [VARIANT_CLASS_LEGEND[code] for code in table.variant_class[: len(RECORDS)]]
        assert classes == ["deletion", "deletion", "insertion", "deletion", "insertion", "deletion", "str_vntr_repeat", "snv", "insertion"]
        assert table.annotations["sv_length"][0] == 600 and table.annotations["sv_length"][8] == 52
        # The deletion's core is [1000, 1600), inside the gene (1500-3500); its padding base alone is not.
        assert table.annotations["in_gene"][: len(RECORDS)].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0]
        np.testing.assert_allclose(table.genetic_position_cm[: len(RECORDS)], [0.002 * record[0] for record in RECORDS], rtol=np.finfo(np.float32).eps)
        assert table.variant_ids(np.arange(len(RECORDS)))[:3] == ["rs0", "chr21:2000:ACGT:A", "rs2"]
        half_b = _written(plain, "chr22", "B", GLIMPSE2.decimals)
        start = store.chromosome_starts[1]
        columns = np.arange(store.n_samples - half_b.shape[1], store.n_samples)
        np.testing.assert_array_equal(store.read_codes(start, start + len(RECORDS), columns), encode_dosage_milli(half_b))


MINIMAC4 = _Tool(
    info_lines=(
        '##INFO=<ID=AF,Number=1,Type=Float,Description="Estimated Alternate Allele Frequency">',
        '##INFO=<ID=R2,Number=1,Type=Float,Description="Estimated Imputation Accuracy (R-square)">',
        '##INFO=<ID=IMPUTED,Number=0,Type=Flag,Description="Marker was imputed">',
    ),
    probabilities=True,
    decimals=3,
    # Batch A/001 reports the last record's R2 above 1, as Minimac4's variance ratio can.
    info=lambda chromosome, batch, record: f"AF=0.3;R2={1.004 if (batch, record) == ('A/001', 8) else _reported_info(record):.5f};IMPUTED",
    identifier=lambda chromosome, record: f"{chromosome}:{RECORDS[record][0]}:{RECORDS[record][1]}:{RECORDS[record][2]}",
)


def test_a_reported_r2_above_one_is_set_to_one_and_counted_and_gp_is_checked_where_written(workspace) -> None:
    """Half A is Minimac4 output with GP, half B the same without GP: the R2 of 1.004 enters the reported r^2 as 1 and is
    counted in the summary and the MANIFEST, and G4 says which batch files it checked."""
    synthetic, client = workspace
    plain = _plain(synthetic, {"A": MINIMAC4, "B": dataclasses.replace(MINIMAC4, probabilities=False)})
    summaries = run_pipeline(WorkspaceConfig.from_mapping(plain.raw), _bindings(_Recorder(), client), BUDGET, through="store")
    gates = summaries["store"]["gates"]
    assert summaries["store"]["reported_r2_above_one"] == len(CHROMOSOMES) and gates["reported_r2"].startswith("TRUNCATED: 2 reported r^2 values above 1")
    assert "largest 1.004" in gates["reported_r2"]
    assert gates["G4_format"] == "NOT_CHECKED on 2 of 6 imputed batch files, which write no FORMAT/GP; PASS on the other 4"
    widths = [width for _label, batches in HALVES for _batch, width in batches]
    reported = np.load(Path(plain.raw["run_directory"]) / "store" / "reported" / "chr21.npy")
    np.testing.assert_allclose(reported[8], (widths[0] * 1.0 + (widths[1] + widths[2]) * _reported_info(8)) / sum(widths), rtol=np.finfo(np.float32).eps)


def _write_vcf(path: Path, info_lines: tuple[str, ...], rows: list[str]) -> Path:
    path.write_text("\n".join([
        "##fileformat=VCFv4.2", *info_lines, FORMAT_LINES["GT"], FORMAT_LINES["DS"], "##contig=<ID=chr21,length=50000000>",
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "S1"]), *rows,
    ]) + "\n")
    return path


def test_a_plain_site_list_refuses_multiallelic_records_and_shared_identifiers(tmp_path: Path) -> None:
    multiallelic = _write_vcf(tmp_path / "multi.vcf", BEAGLE.info_lines, ["chr21\t100\t.\tA\tG\t.\tPASS\tDR2=0.9\tGT:DS\t0|0:0", "chr21\t200\t.\tA\tC,T\t.\tPASS\tDR2=0.9\tGT:DS\t0|0:0"])
    with pytest.raises(ValueError, match=r"multi\.vcf: the record at chr21:200 has 2 ALT alleles.*bcftools norm -m-any"):
        read_plain_sites(multiallelic, "chr21")
    shared = _write_vcf(tmp_path / "shared.vcf", BEAGLE.info_lines, ["chr21\t100\trs7\tA\tG\t.\tPASS\tDR2=0.9\tGT:DS\t0|0:0", "chr21\t100\trs7\tA\tC\t.\tPASS\tDR2=0.9\tGT:DS\t0|0:0"])
    with pytest.raises(ValueError, match="1 identifiers name more than one record, the first 'rs7'"):
        read_plain_sites(shared, "chr21")
    elsewhere = _write_vcf(tmp_path / "elsewhere.vcf", BEAGLE.info_lines, ["22\t100\t.\tA\tG\t.\tPASS\tDR2=0.9\tGT:DS\t0|0:0"])
    with pytest.raises(ValueError, match="lies on 22, not on chr21"):
        read_plain_sites(elsewhere, "chr21")


def test_a_plain_header_declares_exactly_one_reported_r2_field(tmp_path: Path) -> None:
    both = _write_vcf(tmp_path / "both.vcf", (*BEAGLE.info_lines, MINIMAC4.info_lines[1]), ["chr21\t100\t.\tA\tG\t.\tPASS\tDR2=0.9;R2=0.9\tGT:DS\t0|0:0"])
    with pytest.raises(ValueError, match=r"declares the r\^2 fields \['DR2', 'R2'\].*exactly one"):
        plain_batch_fields(both)
    neither = _write_vcf(tmp_path / "neither.vcf", BEAGLE.info_lines[:1], ["chr21\t100\t.\tA\tG\t.\tPASS\tAF=0.1\tGT:DS\t0|0:0"])
    with pytest.raises(ValueError, match=r"declares the r\^2 fields \[\]"):
        plain_batch_fields(neither)


def test_the_config_pairs_the_popped_sidecars_and_names_the_genetic_map_key(workspace) -> None:
    synthetic, _client = workspace
    with pytest.raises(ValueError, match="set together .* or null together"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "strata_directory": None})
    with pytest.raises(ValueError, match=r"plink\.GRCh38\.map\.zip"):
        WorkspaceConfig.from_mapping({key: value for key, value in synthetic.raw.items() if key != "genetic_map"})


def test_a_plain_chromosome_without_cm_or_a_genetic_map_stops_the_store(workspace) -> None:
    synthetic, client = workspace
    plain = _plain(synthetic, {"A": BEAGLE, "B": BEAGLE}, genetic_map=None)
    with pytest.raises(ValueError, match=r"chr21: the batches' records lack INFO/CM.*genetic_map.*plink\.GRCh38\.map\.zip"):
        run_pipeline(WorkspaceConfig.from_mapping(plain.raw), _bindings(_Recorder(), client), BUDGET, through="store")


def test_plain_batches_whose_records_differ_stop_the_store(workspace) -> None:
    """Half B's batch gives one record an SVLEN half A's first batch lacks: the same POS, alleles and ID, but not the
    same record, so its site lines' md5 differs from the site list's."""
    synthetic, client = workspace
    differing = dataclasses.replace(
        BEAGLE,
        info_lines=(*BEAGLE.info_lines, GLIMPSE2.info_lines[5]),
        info=lambda chromosome, batch, record: _beagle_info(chromosome, batch, record) + (";SVLEN=-3" if (batch, record) == ("B/001", 1) else ""),
    )
    plain = _plain(synthetic, {"A": BEAGLE, "B": differing})
    with pytest.raises(ValueError, match=r"B/chr21\.001\.vcf: its records' site lines differ from the site list's"):
        run_pipeline(WorkspaceConfig.from_mapping(plain.raw), _bindings(_Recorder(), client), BUDGET, through="store")
