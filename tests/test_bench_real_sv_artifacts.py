"""Checks of bench-real's SV artifact tools on synthetic inputs only."""

import gzip
import math

import numpy as np

from benchmarks.bench_real import sv_artifacts

EPSILON = np.finfo(np.float64).eps


def exact_hwe_by_enumeration(heterozygotes: int, homozygous_first: int, homozygous_second: int) -> float:
    individuals = heterozygotes + homozygous_first + homozygous_second
    rare = 2 * min(homozygous_first, homozygous_second) + heterozygotes
    common = 2 * individuals - rare

    def log_probability(het):
        rare_hom, common_hom = (rare - het) // 2, individuals - het - (rare - het) // 2
        return (math.lgamma(individuals + 1) - math.lgamma(rare_hom + 1) - math.lgamma(het + 1) - math.lgamma(common_hom + 1)
                + het * math.log(2) + math.lgamma(rare + 1) + math.lgamma(common + 1) - math.lgamma(2 * individuals + 1))

    support = [het for het in range(rare % 2, rare + 1, 2) if (rare - het) // 2 + het <= individuals]
    probabilities = np.exp([log_probability(het) for het in support])
    observed = probabilities[support.index(heterozygotes)]
    return float(probabilities[probabilities <= observed * (1 + 1e3 * EPSILON)].sum())


def test_hwe_exact_matches_enumeration():
    for counts in [(10, 5, 5), (0, 10, 10), (30, 60, 10), (1, 99, 0), (57, 40, 3), (2, 0, 98)]:
        assert abs(sv_artifacts.hwe_exact(*counts) - exact_hwe_by_enumeration(*counts)) < 1e6 * EPSILON


def test_mendelian_consistency_follows_transmission():
    assert sv_artifacts.mendelian_consistent(1, 0, 2)
    assert not sv_artifacts.mendelian_consistent(0, 0, 2)
    assert not sv_artifacts.mendelian_consistent(2, 1, 0)
    assert sv_artifacts.mendelian_consistent(2, 1, 1) and sv_artifacts.mendelian_consistent(0, 1, 1)


def test_minor_carriers_follow_the_minor_allele():
    assert list(sv_artifacts.minor_carriers(np.array([0, 0, 1, 2, 0]))) == [False, False, True, True, False]
    assert list(sv_artifacts.minor_carriers(np.array([2, 2, 1, 0, 2]))) == [False, False, True, True, False]


def test_cross_mappable_partners_and_gencode_parse(tmp_path):
    crossmap = tmp_path / "cross.txt.gz"
    with gzip.open(crossmap, "wt") as handle:
        handle.write("ENSG1.1\tENSG2.3\t5\nENSG2.3\tENSG3.1\t1.5\nENSG4.1\tENSG5.1\t2\n")
    partners = sv_artifacts.cross_mappable_partners(crossmap, {"ENSG2", "ENSG5"})
    assert partners == {"ENSG2": {"ENSG1": 5.0, "ENSG3": 1.5}, "ENSG5": {"ENSG4": 2.0}}
    gtf = tmp_path / "genes.gtf.gz"
    with gzip.open(gtf, "wt") as handle:
        handle.write("#header\n")
        handle.write('chr1\tHAVANA\tgene\t100\t200\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "x"; gene_name "ONE";\n')
        handle.write('chr1\tHAVANA\texon\t100\t150\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "ONE";\n')
    genes = sv_artifacts.gencode_genes(gtf)
    assert list(genes.index) == ["ENSG1"] and genes.loc["ENSG1", "name"] == "ONE" and genes.loc["ENSG1", "end"] == 200


def test_residual_is_orthogonal_to_the_design():
    generator = np.random.default_rng(0)
    design, values = generator.normal(size=(50, 3)), generator.normal(size=50)
    remainder = sv_artifacts.residual(values, design)
    assert np.allclose(np.column_stack([np.ones(50), design]).T @ remainder, 0, atol=1e3 * EPSILON)
