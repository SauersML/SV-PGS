"""Are SV effects on their gene's expression sign-skewed? [real, bench-real training people only]

For each gene of genes500 and its loso/AFR training people (covariate-residualized expression, as the harness gives
it), every polymorphic SV of class DEL or INS/DUP (ALT = presence of the event) gets its marginal association z with
the gene's expression. Statistics per class x overlap (gene body, exon, neither): the mean z, and among |z| > 1.96 the
share of negative signs. Null: expression permuted across genes (each window scored against another gene's
expression), which keeps the windows' LD and the expression's own distribution; 2000 permutations."""
import sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from benchmarks.bench_real import harness  # noqa: E402

dataset = harness.Dataset("/scratch.global/sauer354/svpgs-team/bench-real/dataset")
split = dataset.splits["loso/AFR"]
chromosomes = [f"chr{n}" for n in range(1, 23)]
rows = dataset.gene_rows(chromosomes, None, "/scratch.global/sauer354/svpgs-team/lead/genes500.tsv")
windows, phenotypes = [], []
for gene_row in rows:
    window = harness.load_gene_window(dataset, int(gene_row))
    train, _test_genotypes, _test_phenotype, _test_index = harness.build_gene_task(dataset, window, split)
    variants = train.variants
    base = np.array([str(value).split(":")[0] for value in variants.sv_type])
    klass = np.where(base == "DEL", "DEL", np.where(np.isin(base, ["INS", "DUP"]), "INS/DUP", ""))
    keep = np.asarray(variants.is_sv, dtype=bool) & (klass != "")
    if not keep.any():
        continue
    start, end = np.asarray(variants.position)[keep], np.asarray(variants.end)[keep]
    in_body = (end >= train.gene_start) & (start <= train.gene_end)
    in_exon = np.zeros(start.shape[0], dtype=bool)
    for exon_start, exon_end in train.exons:
        in_exon |= (end >= exon_start) & (start <= exon_end)
    overlap = np.where(in_exon, "exon", np.where(in_body, "body", "outside"))
    genotypes = np.asarray(train.genotypes, dtype=np.float64)[:, keep]
    genotypes = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    windows.append((genotypes, klass[keep], overlap))
    phenotypes.append(np.asarray(train.phenotype, dtype=np.float64))
count = phenotypes[0].shape[0]
assert all(p.shape[0] == count for p in phenotypes)
standardized = [(p - p.mean()) / p.std() for p in phenotypes]
print("genes with SVs", len(windows), "SVs", sum(w[0].shape[1] for w in windows), "training people", count, flush=True)


def statistics(order):
    z, labels = [], []
    for (genotypes, klass, overlap), index in zip(windows, order):
        correlation = genotypes.T @ standardized[index] / count
        z.append(correlation * np.sqrt(count - 2) / np.sqrt(np.maximum(1.0 - correlation**2, 1e-300)))
        labels.append(np.char.add(np.char.add(klass.astype(str), "|"), overlap.astype(str)))
    return np.concatenate(z), np.concatenate(labels)


observed_z, labels = statistics(range(len(windows)))
groups = {"all": np.ones(labels.shape[0], dtype=bool)}
for klass in ("DEL", "INS/DUP"):
    groups[klass] = np.char.startswith(labels, klass + "|")
    for overlap in ("exon", "body", "outside"):
        groups[f"{klass} {overlap}"] = labels == f"{klass}|{overlap}"


def summaries(z):
    out = {}
    for name, mask in groups.items():
        strong = mask & (np.abs(z) > 1.96)
        out[name] = (z[mask].mean() if mask.any() else np.nan, (z[strong] < 0).mean() if strong.any() else np.nan, int(mask.sum()), int(strong.sum()))
    return out


observed = summaries(observed_z)
rng = np.random.default_rng(0)
null = []
for _ in range(2000):
    order = rng.permutation(len(windows))
    while np.any(order == np.arange(len(windows))):
        order = rng.permutation(len(windows))
    null.append(summaries(statistics(order)[0]))
for name in groups:
    mean_z, negative_share, size, strong = observed[name]
    null_mean = np.array([entry[name][0] for entry in null])
    null_share = np.array([entry[name][1] for entry in null])
    p_mean = (1 + np.sum(np.abs(null_mean - np.nanmean(null_mean)) >= abs(mean_z - np.nanmean(null_mean)))) / (1 + null_mean.size)
    finite = np.isfinite(null_share)
    p_share = (1 + np.sum(np.abs(null_share[finite] - np.nanmean(null_share)) >= abs(negative_share - np.nanmean(null_share)))) / (1 + finite.sum())
    low, high = np.nanpercentile(null_share, [2.5, 97.5])
    print(f"{name:>18}: SVs {size:5d} strong {strong:4d} mean z {mean_z:+.3f} (null 95% [{np.percentile(null_mean, 2.5):+.3f}, {np.percentile(null_mean, 97.5):+.3f}], p {p_mean:.4f})"
          f"  negative share among |z|>1.96 {negative_share:.3f} (null 95% [{low:.3f}, {high:.3f}], p {p_share:.4f})", flush=True)

gene_of = np.concatenate([np.full(w[0].shape[1], index) for index, w in enumerate(windows)])
for name in ("DEL exon", "INS/DUP exon", "DEL body", "INS/DUP body"):
    mask = groups[name]
    genes = np.unique(gene_of[mask])
    per_gene = np.array([observed_z[mask & (gene_of == gene)].mean() for gene in genes])
    boots = [per_gene[rng.integers(0, genes.size, genes.size)].mean() for _ in range(10000)]
    print(f"{name}: genes {genes.size}, gene-mean z {per_gene.mean():+.3f} gene-bootstrap 95% CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}], genes with mean z < 0: {int((per_gene < 0).sum())}", flush=True)
