"""The benchmark seed helper: a seed is a function of a whole name, never of its first bytes."""
import pathlib
import subprocess
import sys

from benchmarks import seeds

# Three real Ensembl gene ids (TP53, BRCA2, BRCA1). Every Ensembl human gene id begins "ENSG0000", so a seed taken
# from the first eight bytes is one seed for the whole benchmark.
REAL_GENE_IDS = ("ENSG00000141510", "ENSG00000139618", "ENSG00000012048")
REPOSITORY_ROOT = pathlib.Path(seeds.__file__).resolve().parent.parent


def test_real_gene_ids_sharing_a_prefix_get_different_seeds():
    assert len({name.encode()[:8] for name in REAL_GENE_IDS}) == 1
    assert len({seeds.seed_from_name(name) for name in REAL_GENE_IDS}) == len(REAL_GENE_IDS)


def test_the_seed_is_stable_across_processes():
    """PYTHONHASHSEED differs per process, so a seed from hash() would not repeat between runs; this one does."""
    script = f"from benchmarks import seeds; print(' '.join(str(seeds.seed_from_name(n)) for n in {REAL_GENE_IDS!r}))"
    printed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                             cwd=REPOSITORY_ROOT).stdout.split()
    assert [int(value) for value in printed] == [seeds.seed_from_name(name) for name in REAL_GENE_IDS]


def test_the_benchmarks_seed_their_per_name_fits_through_the_helper():
    """The fit that seeds from a name (bench-real's small-n route) calls the helper."""
    from benchmarks import svpgs_small_n

    assert svpgs_small_n.seed_from_name is seeds.seed_from_name


def test_no_benchmark_builds_a_seed_from_a_name_prefix():
    sources = sorted((REPOSITORY_ROOT / "benchmarks").rglob("*.py"))
    assert sources, "the benchmark sources were not found"
    assert [str(path) for path in sources if "encode()[:" in path.read_text()] == []
