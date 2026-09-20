"""The sealed confirmation compounds of the DREAM cytotoxicity benchmark.

A compound is sealed when sha256("bench-tox/confirm/" + its NCGC id) is 0 mod 4. The rule depends only on the id, so
it was fixed before any value was looked at. Sealed compounds are scored once, at the lead's call, and no
development analysis (heritability, method choice, adoption) may read them.
"""
import hashlib


def sealed(compound: str) -> bool:
    return int(hashlib.sha256(f"bench-tox/confirm/{compound}".encode()).hexdigest(), 16) % 4 == 0


def split_compounds(compounds):
    """(development, confirmation), each in the given order."""
    return [compound for compound in compounds if not sealed(compound)], [compound for compound in compounds if sealed(compound)]
