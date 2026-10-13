"""Check declared out-of-reference-panel validation manifests without exposing identities.

This checks manifest consistency and binds the imputed files. It cannot independently
prove that an external imputation job used the declared reference.
"""
from __future__ import annotations

from pathlib import Path
from sv_pgs.content_identity import file_digest


def validate_reference_exclusion(manifest, validation_ids, imputed_files):
    if manifest.get('schema') != 'svpgs-reference-exclusion-v1':
        raise ValueError('Unknown reference-exclusion manifest schema')
    for field in ('genome_build', 'reference_callset', 'family_definition'):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f'Reference-exclusion manifest needs {field}')
    roles = {}
    for role in ('reference_people', 'validation_people'):
        rows = manifest.get(role, [])
        if not rows or any(not row.get('id') or not row.get('family') for row in rows):
            raise ValueError(f'{role} needs person and globally consistent family IDs')
        roles[role] = {str(row['id']): str(row['family']) for row in rows}
        if len(roles[role]) != len(rows):
            raise ValueError(f'{role} repeats a person')
    reference, validation = roles['reference_people'], roles['validation_people']
    if set(reference) & set(validation) or set(reference.values()) & set(validation.values()):
        raise ValueError('Imputation validation people or close relatives overlap the reference')
    if not set(map(str, validation_ids)) <= set(validation):
        raise ValueError('Manifest does not cover all truth-calibration people')
    declared = manifest.get('imputed_sha256', {})
    for path in imputed_files:
        if declared.get(str(Path(path))) != file_digest(path):
            raise ValueError('Imputed input is not bound to the reference-exclusion manifest')
    return {'status': 'manifest_consistent', 'external_imputation_execution_verified': False,
            'reference_callset': manifest['reference_callset'], 'genome_build': manifest['genome_build']}
