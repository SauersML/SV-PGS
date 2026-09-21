"""One seed derivation for the benchmarks: a seed is a function of the whole name.

A prefix of a name is not enough to tell names apart. Ensembl gene ids all begin "ENSG0000", so a seed read from a
gene id's first eight bytes is the same seed for every gene, and a run that means to draw independently per gene
draws the same numbers each time. The digest covers the whole string instead, so distinct names give distinct seeds
apart from the 2^-64 collision rate of a 64-bit digest.

sha256 of the text is also a pure function of the text: unlike ``hash()``, which PYTHONHASHSEED randomizes per
process, it gives the same seed in every process and every run, so a benchmark result is reproducible.
"""
import hashlib


def seed_from_name(name: str) -> int:
    """The 64-bit seed of ``name``, from a sha256 of the whole string."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
