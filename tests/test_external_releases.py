import gzip

import numpy as np
import pytest

from sv_pgs.external_annotations import (
    annotate_records,
    keyed_snv_associations,
    lift_positions,
    read_bai_structural,
    read_bai_tandem_repeat,
    read_panukb_eur,
)


def _write(path, lines):
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def test_bai_releases_give_squared_z_keyed_by_their_ids(tmp_path):
    sv = _write(tmp_path / "trait.SV.gz", [
        "CHR\tSNP\tPOS\tA1\tA2\tN\tAF1\tBETA\tSE\tP\tMAF",
        "1\tchr1_HQA241SV_7\t20851\tAGTC\tA\t400000\t0.2\t0.03\t0.01\t0.003\t0.2",
        "2\tchr2_HQA241SV_9\t5000\tG\tGTTTT\t400000\t0.4\t-0.02\t0.005\t0.0001\t0.4",
    ])
    vntr = _write(tmp_path / "trait.VNTR.gz", [
        "ID\tVNTR_locus\tChromosome\tStart\tEnd\tRepeat_unit\tBETA\tSE\tP\tN\tPercent_of_matches\tPeriod_size\tPercent_of_ACGT_composition",
        "HQA241_HG38_chr1_100_190_15\t1:100-190\t1\t100\t190\tTATATTTTATATATA\t-0.004\t0.002\t0.05\t330000\t92\t15\t41:0:1:57",
    ])
    structural = read_bai_structural(sv)
    np.testing.assert_array_equal(structural.keys, ["chr1_HQA241SV_7", "chr2_HQA241SV_9"])
    np.testing.assert_allclose(structural.squared_z, [9.0, 16.0])
    repeat = read_bai_tandem_repeat(vntr)
    np.testing.assert_allclose(repeat.squared_z, [4.0])


def test_duplicate_or_invalid_rows_fail_loudly(tmp_path):
    header = "CHR\tSNP\tPOS\tA1\tA2\tN\tAF1\tBETA\tSE\tP\tMAF"
    duplicated = _write(tmp_path / "dup.SV.gz", [header, "1\tsv1\t1\tA\tT\t1\t0.1\t0.1\t0.1\t0.5\t0.1",
                                                  "1\tsv1\t1\tA\tT\t1\t0.1\t0.1\t0.1\t0.5\t0.1"])
    with pytest.raises(ValueError, match="more than once"):
        read_bai_structural(duplicated)
    zero_se = _write(tmp_path / "zero.SV.gz", [header, "1\tsv1\t1\tA\tT\t1\t0.1\t0.1\t0\t0.5\t0.1"])
    with pytest.raises(ValueError, match="positive standard errors"):
        read_bai_structural(zero_se)


def test_panukb_lift_and_join_keep_only_confident_mapped_sites(tmp_path):
    release = _write(tmp_path / "trait.EUR.tsv.gz", [
        "chr\tpos\tref\talt\tbeta_EUR\tse_EUR\tneglog10_pval_EUR\taf_EUR\tlow_confidence_EUR",
        "1\t1000\tA\tG\t0.2\t0.1\t1.3\t0.3\tfalse",
        "1\t2000\tC\tT\t0.3\t0.1\t2.0\t0.2\ttrue",
        "1\t3000\tG\tA\t-0.1\t0.05\t1.3\t0.1\tfalse",
        "X\t4000\tG\tA\t0.5\t0.1\t5.0\t0.4\tfalse",
    ])
    associations = read_panukb_eur(release)
    np.testing.assert_array_equal(associations.position, [1000, 3000])
    lifted, mapped = lift_positions(associations.chromosome, associations.position, [1, 1], [1000, 5000], [1100, 5100])
    np.testing.assert_array_equal(lifted, [1100, -1])
    np.testing.assert_array_equal(mapped, [True, False])
    keyed = keyed_snv_associations(associations, lifted, mapped)
    np.testing.assert_array_equal(keyed.keys, ["1:1100:A:G"])
    annotation = annotate_records(np.array(["1:1100:A:G", "1:3000:G:A", "1:1100:A:T"], dtype=object), keyed)
    np.testing.assert_array_equal(annotation.present, [True, False, False])
    np.testing.assert_allclose(annotation.log_squared_z, [np.log1p(4.0), 0.0, 0.0])


def test_lift_positions_rejects_empty_or_ambiguous_maps():
    with pytest.raises(ValueError, match="empty"):
        lift_positions([1], [10], [], [], [])
    with pytest.raises(ValueError, match="more than once"):
        lift_positions([1], [10], [1, 1], [10, 10], [11, 12])
