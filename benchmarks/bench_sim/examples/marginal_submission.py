"""Harness smoke submission: marginal regression scores on standardized dosages, shrunk as one infinitesimal block.

It only exercises the submission API; it is not a baseline.
"""

import numpy as np

CODES_PER_DOSAGE = 127.0
BLOCK = 8192


class Model:
    def __init__(self, rows, mean, sd, beta, structural):
        self.rows, self.mean, self.sd, self.beta, self.structural = rows, mean, sd, beta, structural

    def score(self, test):
        total = np.zeros(test.n_samples)
        structural = np.zeros(test.n_samples)
        for first in range(0, self.rows.size, BLOCK):
            rows = self.rows[first:first + BLOCK]
            standardized = (test.codes(rows) / CODES_PER_DOSAGE - self.mean[first:first + BLOCK, None]) / self.sd[first:first + BLOCK, None]
            total += self.beta[first:first + BLOCK] @ standardized
            members = self.structural[first:first + BLOCK]
            structural += self.beta[first:first + BLOCK][members] @ standardized[members]
        return {"total": total, "structural": structural}


def fit(train):
    design = np.column_stack([np.ones(train.n_samples), train.covariates])
    coefficients, *_ = np.linalg.lstsq(design, train.phenotype, rcond=None)
    residual = train.phenotype - design @ coefficients
    residual /= residual.std()
    kept, means, sds, betas = [], [], [], []
    for first in range(0, train.n_variants, BLOCK):
        rows = np.arange(first, min(first + BLOCK, train.n_variants))
        dosage = train.codes(rows) / CODES_PER_DOSAGE
        mean, sd = dosage.mean(axis=1), dosage.std(axis=1)
        live = sd > 0
        marginal = ((dosage[live] - mean[live, None]) / sd[live, None]) @ residual / train.n_samples
        kept.append(rows[live])
        means.append(mean[live])
        sds.append(sd[live])
        betas.append(marginal)
    rows = np.concatenate(kept)
    beta = np.concatenate(betas) / rows.size
    return Model(rows, np.concatenate(means), np.concatenate(sds), beta, train.variants["cls"][rows] >= 2)
