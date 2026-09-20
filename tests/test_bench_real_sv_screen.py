"""Math and window checks of the bench-real SV screen on synthetic data (correctness only, never accuracy evidence)."""
import json

import numpy as np
import pandas as pd

from benchmarks.bench_real import harness, sv_screen

EPSILON = np.finfo(np.float64).eps


def synthetic_chromosome(seed, samples=120, small=400, panel_sv=30, pangenie_sv=12, span=30_000):
    """A panel section sorted by position (SVs interleaved) followed by PanGenie SVs, as build_dataset writes them."""
    generator = np.random.default_rng(seed)
    small_positions = np.sort(generator.integers(1, span, small))
    small_ends = small_positions + generator.integers(0, 49, small)
    small_genotypes = generator.binomial(2, generator.uniform(0.02, 0.5, small)[:, None], size=(small, samples))
    small_genotypes[0] = 1
    sv_positions = np.sort(generator.integers(1, span, panel_sv + pangenie_sv))
    sv_ends = sv_positions + generator.integers(50, 3_000, panel_sv + pangenie_sv)
    sv_genotypes = generator.binomial(2, generator.uniform(0.02, 0.5, panel_sv + pangenie_sv)[:, None], size=(panel_sv + pangenie_sv, samples))
    for index in range(0, panel_sv + pangenie_sv, 3):
        nearest = int(np.argmin(np.abs(small_positions - sv_positions[index])))
        noisy = generator.random(samples) < 0.1
        sv_genotypes[index] = np.where(noisy, generator.integers(0, 3, samples), small_genotypes[nearest])
    sv_genotypes[1] = 2
    sv_genotypes[4] = 1
    source = generator.permutation(np.array(["panel"] * panel_sv + ["pangenie"] * pangenie_sv))
    small_table = pd.DataFrame({"pos": small_positions, "end": small_ends, "is_sv": False, "source": "panel", "sv_type": ".", "sv_length": 0})
    sv_table = pd.DataFrame({"pos": sv_positions, "end": sv_ends, "is_sv": True, "source": source,
                             "sv_type": generator.choice(["DEL", "DUP", "INS"], panel_sv + pangenie_sv), "sv_length": sv_ends - sv_positions})
    table = pd.concat([small_table, sv_table], ignore_index=True)
    genotypes = np.concatenate([small_genotypes, sv_genotypes])
    panel = table[table["source"] == "panel"].sort_values("pos", kind="stable")
    pangenie = table[table["source"] == "pangenie"].sort_values("pos", kind="stable")
    rows = np.concatenate([panel.index.to_numpy(), pangenie.index.to_numpy()])
    table = table.loc[rows].reset_index(drop=True)
    table["id"] = [f"variant{index}" for index in range(len(table))]
    table["ref_len"], table["alt_len"], table["symbolic"] = 1, 1, False
    return table, genotypes[rows].astype(np.int8)


def brute_force_proxies(table, dosage, radius):
    small = np.flatnonzero((table["source"] == "panel").to_numpy() & ~table["is_sv"].to_numpy())
    result = {}
    for row in np.flatnonzero(table["is_sv"].to_numpy()):
        genotype = dosage[row].astype(np.float64)
        low, high = table["pos"].iloc[row] - radius, table["end"].iloc[row] + radius
        best = 0.0
        for other in small:
            if table["end"].iloc[other] >= low and table["pos"].iloc[other] <= high:
                proxy = dosage[other].astype(np.float64)
                if genotype.var() > 0 and proxy.var() > 0:
                    best = max(best, np.corrcoef(genotype, proxy)[0, 1] ** 2)
        result[row] = (genotype.var(), best, genotype.var() * (1 - best))
    return result


def check_against_brute_force(table, dosage, radius, **options):
    frame = sv_screen.sv_proxies("chr1", table, dosage, radius=radius, **options)
    expected = brute_force_proxies(table, dosage, radius)
    tolerance = dosage.shape[1] * EPSILON
    assert sorted(frame["row"]) == sorted(expected)
    for row in frame.itertuples():
        variance, best, untagged = expected[row.row]
        assert abs(row.genotype_variance - variance) <= tolerance
        assert abs(row.max_r2 - best) <= tolerance
        assert abs(row.untagged_variance - untagged) <= tolerance


def test_proxies_match_a_direct_computation_in_one_chunk_and_in_many():
    table, dosage = synthetic_chromosome(seed=1)
    check_against_brute_force(table, dosage, radius=1_500, chunk_rows=len(table))
    check_against_brute_force(table, dosage, radius=1_500, chunk_rows=7)


def test_an_sv_longer_than_the_radius_is_its_own_group_and_still_exact():
    table, dosage = synthetic_chromosome(seed=5)
    longest = int(np.flatnonzero(table["is_sv"].to_numpy())[3])
    table.loc[longest, "end"] = int(table["pos"].max())
    groups = list(sv_screen._batches(np.array([0, 10, 20, 30]), np.array([5, 5_000, 25, 35]), radius=100))
    assert [group.tolist() for group in groups] == [[0], [1], [2, 3]]
    check_against_brute_force(table, dosage, radius=800, chunk_rows=11)


def test_the_memory_sized_chunks_give_the_same_answer():
    table, dosage = synthetic_chromosome(seed=6)
    exact = sv_screen.sv_proxies("chr1", table, dosage, radius=1_000, chunk_rows=len(table))
    sized = sv_screen.sv_proxies("chr1", table, dosage, radius=1_000, worker_bytes=sv_screen.resident_bytes() + (8 << 20))
    pd.testing.assert_frame_equal(exact, sized)


def test_constant_svs_carry_no_untagged_variance_and_constant_proxies_never_tag():
    table, dosage = synthetic_chromosome(seed=2)
    frame = sv_screen.sv_proxies("chr1", table, dosage, radius=2_000, chunk_rows=len(table))
    constant = frame[frame["genotype_variance"] == 0]
    assert len(constant) >= 2
    assert (constant["untagged_variance"] == 0).all() and (constant["max_r2"] == 0).all()
    constant_proxy = int(np.flatnonzero((dosage.var(axis=1) == 0) & ~table["is_sv"].to_numpy())[0])
    assert constant_proxy not in set(frame["proxy_row"])


def test_standardization_matches_numpy():
    generator = np.random.default_rng(3)
    dosage = generator.integers(0, 3, size=(200, 50)).astype(np.int8)
    dosage[7] = 1
    rows = np.sort(generator.choice(200, 120, replace=False))
    values = dosage[rows].astype(np.float64)
    spread = values.std(axis=1)
    expected = np.where(spread[:, None] > 0, (values - values.mean(axis=1, keepdims=True)) / np.where(spread > 0, spread, 1)[:, None], 0.0)
    np.testing.assert_allclose(sv_screen.read_standardized(dosage, rows), expected, rtol=0, atol=dosage.shape[1] * EPSILON)


def write_dataset(directory, table, dosage, genes, annotation):
    samples = dosage.shape[1]
    pd.DataFrame({"sample": [f"s{index}" for index in range(samples)], "Superpopulation": "EUR", "Population": "GBR"}).to_csv(
        directory / "samples.tsv", sep="\t", index=False)
    genes.to_csv(directory / "genes.tsv", sep="\t", index=False)
    np.save(directory / "expression.npy", np.zeros((len(genes), samples)))
    np.save(directory / "covariates.npy", np.zeros((samples, 1)))
    (directory / "splits.json").write_text(json.dumps([]))
    (directory / "gene_annotation.json").write_text(json.dumps(annotation))
    table.to_csv(directory / "chr1.variants.tsv", sep="\t", index=False)
    np.save(directory / "chr1.dosage.npy", dosage)


def test_gene_windows_match_the_harness_and_sums_match_the_proxies(tmp_path):
    table, dosage = synthetic_chromosome(seed=4)
    radius = harness.CIS_RADIUS_BP
    scaled = table.assign(pos=table["pos"] * 100, end=table["end"] * 100)
    tss = np.array([200_000, 1_500_000, 2_900_000])
    genes = pd.DataFrame({"chrom": "chr1", "start": tss - 1, "end": tss, "gene_id": ["g0", "g1", "g2"], "tss": tss})
    annotation = {gene: {"start": int(start), "end": int(start) + 50_000, "strand": "+", "exons": [[int(start), int(start) + 200]],
                         "coding_exons": []} for gene, start in zip(genes["gene_id"], tss)}
    write_dataset(tmp_path, scaled, dosage, genes, annotation)
    dataset = harness.Dataset(tmp_path)
    frame = sv_screen.sv_proxies("chr1", scaled, dosage, radius=radius, chunk_rows=5)
    scores = sv_screen.gene_scores(genes, annotation, frame, radius=radius).set_index("gene_id")
    for gene in genes.itertuples():
        cis = dataset.cis_rows("chr1", int(gene.tss))
        structural = set(cis[scaled["is_sv"].to_numpy()[cis] & (dosage[cis].var(axis=1) > 0)])
        members = sv_screen.window_members(frame["pos"].to_numpy(), frame["end"].to_numpy(), int(gene.tss), radius)
        counted = set(frame["row"].to_numpy()[members][frame["genotype_variance"].to_numpy()[members] > 0])
        assert counted == structural
        chosen = frame[frame["row"].isin(structural)]
        for source, name in (("panel", "panel"), ("pangenie", "pgsv")):
            mine = chosen[chosen["source"] == source]
            assert scores.loc[gene.gene_id, f"n_sv_{name}"] == len(mine)
            assert abs(scores.loc[gene.gene_id, f"U_{name}"] - mine["untagged_variance"].sum()) <= len(mine) * EPSILON


def test_body_and_exon_overlap_flags():
    frame = pd.DataFrame({"chrom": "chr1", "row": [0, 1], "source": "panel", "pos": [1_000, 5_000],
                          "end": [1_100, 5_010], "sv_type": "DEL", "sv_length": [100, 10], "allele_frequency": 0.2,
                          "genotype_variance": 0.3, "max_r2": 0.1, "proxy_row": -1, "proxy_pos": -1, "proxies": 1, "untagged_variance": [0.27, 0.27]})
    genes = pd.DataFrame({"chrom": "chr1", "gene_id": ["exonic", "intronic", "outside"], "tss": [1_000, 5_000, 20_000]})
    annotation = {"exonic": {"start": 900, "end": 3_000, "exons": [[1_050, 1_060]]},
                  "intronic": {"start": 4_000, "end": 9_000, "exons": [[4_000, 4_100], [8_000, 9_000]]},
                  "outside": {"start": 20_000, "end": 30_000, "exons": [[20_000, 21_000]]}}
    scores = sv_screen.gene_scores(genes, annotation, frame, radius=500).set_index("gene_id")
    assert scores.loc["exonic", "exon_overlap_panel"] and scores.loc["exonic", "body_overlap_panel"]
    assert scores.loc["intronic", "body_overlap_panel"] and not scores.loc["intronic", "exon_overlap_panel"]
    assert scores.loc["outside", "n_sv_panel"] == 0 and scores.loc["outside", "U_panel"] == 0


def test_ranking_orders_by_the_key_and_breaks_ties_by_the_sealed_gene_order():
    scores = pd.DataFrame({"gene_id": ["a", "b", "c", "d"], "U_panel": [0.1, 0.5, 0.1, 0.0], "max_u_panel": [0.3, 0.1, 0.2, 0.3]})
    ranked = sv_screen.rank_genes(scores, ["d", "c", "b", "a"])
    assert ranked["gene_id"].tolist() == ["b", "c", "a", "d"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    assert sv_screen.rank_genes(ranked, ["d", "c", "b", "a"], key="max_u_panel")["gene_id"].tolist() == ["d", "a", "c", "b"]


def test_reorder_keeps_the_genes_marks_dev_refuses_a_changed_source_and_never_replaces(tmp_path):
    import pytest
    screen = pd.DataFrame({"rank": [1, 2, 3, 4], "gene_id": ["a", "b", "c", "d"], "U_panel": [0.5, 0.4, 0.3, 0.2],
                           "max_u_panel": [0.1, 0.4, 0.3, 0.2], "gene_order_index": [3, 2, 1, 0],
                           "already_scored": [True, False, False, True], "confirm": [False, False, True, False]})
    screen.to_csv(tmp_path / "screen_v1.tsv", sep="\t", index=False)
    (tmp_path / "SEALED.txt").write_text(f"{sv_screen._sha256(tmp_path / 'screen_v1.tsv')}  screen_v1.tsv\n")
    ranked = sv_screen.reorder(tmp_path, "v1", "v2", "max_u_panel")
    assert ranked["gene_id"].tolist() == ["b", "c", "d", "a"]
    assert ranked["dev"].tolist() == [False, False, True, True]
    assert pd.read_csv(tmp_path / "sv_ranked_v2.tsv", sep="\t")["gene_id"].tolist() == ["b", "d", "a"]
    assert "sv_ranked_v2.tsv" in (tmp_path / "SEALED.txt").read_text()
    with pytest.raises(RuntimeError):
        sv_screen.reorder(tmp_path, "v1", "v2", "max_u_panel")
    (tmp_path / "screen_v1.tsv").write_text("changed")
    with pytest.raises(RuntimeError):
        sv_screen.reorder(tmp_path, "v1", "v3", "max_u_panel")


def test_confirmation_genes_follow_the_sealed_hash_and_exclude_every_scored_gene():
    genes = [f"ENSG{index:011d}.1" for index in range(4_000)]
    development = genes[:1_000]
    confirm = sv_screen.confirmation_genes(genes, development)
    assert not confirm[:1_000].any()
    import hashlib
    for gene, flag in zip(genes[1_000:1_050], confirm[1_000:1_050]):
        assert flag == (int(hashlib.sha256(("bench-real/confirm/" + gene).encode()).hexdigest(), 16) % 4 == 0)
    share = confirm[1_000:].mean()
    assert abs(share - 0.25) <= 4 * np.sqrt(0.25 * 0.75 / 3_000)
