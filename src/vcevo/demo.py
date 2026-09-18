"""Small synthetic contract fixture; deliberately not a biological benchmark."""
import numpy as np

from .types import Condition, Observation, PreparedDataset


def synthetic_dataset(seed=17):
    rng = np.random.default_rng(seed)
    genes = tuple("GENE_" + str(i) for i in range(12))
    control = np.ones(len(genes), dtype=float)
    observations = {}
    for i in range(16):
        condition = Condition("synthetic_" + str(i), "SYNTHETIC", (genes[i % len(genes)],))
        delta = rng.normal(0, 0.1 + i / 60, len(genes))
        labels = np.where(np.abs(delta) > 0.15, np.sign(delta), 0).astype(np.int8)
        qvalues = np.where(labels == 0, 0.8, 0.01)
        cells = np.maximum(0, control + delta + rng.normal(0, 0.02, (8, len(genes))))
        observations[condition.id] = Observation(condition, delta, labels, qvalues, cells)
    ids = list(observations)
    return PreparedDataset(genes, control, observations,
                           {"initial": ids[:4], "pool": ids[4:12], "validation": [], "test": ids[12:]},
                           {"synthetic": True, "purpose": "contract tests only; not measured biology", "seed": seed})
