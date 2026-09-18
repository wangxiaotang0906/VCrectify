"""Explicit diagnostic baselines, never substitutes labelled TxPert or SUMMER."""
from __future__ import annotations

import hashlib
import numpy as np

from ..types import EvidenceItem, KnowledgePrediction, KnowledgeState


class RidgeBackbone:
    """A deterministic hashed-target ridge regressor for CPU framework checks.

    Hashed features contain no biological knowledge. This is an engineering
    reference, not a competitive perturbation model.
    """
    name = "reference_hashed_ridge"

    def __init__(self, genes, seed=0, feature_dim=16, regularization=1.0):
        if feature_dim < 1 or regularization <= 0:
            raise ValueError("Invalid ridge configuration")
        self.genes = tuple(genes)
        self.seed, self.feature_dim, self.regularization = seed, feature_dim, regularization
        self.weights = np.zeros((feature_dim + 1, len(genes)))

    def _features(self, conditions):
        rows = []
        for c in conditions:
            vectors = []
            for gene in c.targets:
                key = hashlib.sha256((str(self.seed) + ":" + gene + ":" + c.modality).encode()).digest()
                rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
                vectors.append(rng.normal(size=self.feature_dim) / np.sqrt(self.feature_dim))
            rows.append(np.r_[1.0, np.mean(vectors, axis=0)])
        return np.asarray(rows)

    def _solve(self, groups):
        lhs = np.eye(self.feature_dim + 1) * self.regularization
        rhs = np.zeros_like(self.weights)
        for observations, weight in groups:
            if not observations or weight == 0:
                continue
            x = self._features([o.condition for o in observations])
            y = np.stack([o.delta for o in observations])
            lhs += weight * x.T @ x / len(observations)
            rhs += weight * x.T @ y / len(observations)
        self.weights = np.linalg.solve(lhs, rhs)

    def fit(self, observations, control_mean):
        if not observations:
            raise ValueError("Ridge fitting needs observed conditions")
        self._solve([(observations, 1.0)])

    def predict(self, conditions, control_mean):
        return self._features(conditions) @ self.weights

    def update(self, observations, replay, control_mean, eta):
        if not observations:
            return
        self._solve([(observations, eta if replay else 1.0), (replay, 1-eta)])

    def fresh(self, seed):
        return RidgeBackbone(self.genes, seed, self.feature_dim, self.regularization)

    def state_dict(self):
        return {"weights": self.weights, "seed": self.seed,
                "genes": self.genes, "feature_dim": self.feature_dim,
                "regularization": self.regularization}

    def load_state_dict(self, state):
        if (tuple(state.get("genes", ())) != self.genes or
                state.get("feature_dim") != self.feature_dim or
                state.get("regularization") != self.regularization):
            raise ValueError("Ridge checkpoint configuration or gene axis mismatch")
        if np.asarray(state["weights"]).shape != self.weights.shape:
            raise ValueError("Ridge checkpoint gene axis mismatch")
        self.weights = np.asarray(state["weights"]).copy()
        self.seed = state["seed"]


REFERENCE_ITEM = "reference:observed-class-frequency"


def reference_knowledge():
    return KnowledgeState({REFERENCE_ITEM: EvidenceItem(
        REFERENCE_ITEM, "Diagnostic baseline: predict from observed same-context response-class frequencies.",
        (), "vcevo:reference-baseline:v1", kind="algorithm")})


class MemoryReasoner:
    """Observed-label frequency baseline. No LLM or mechanistic explanation."""
    name = "reference_observed_frequency"

    def fresh(self, seed):
        return MemoryReasoner()

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Reference reasoner has no mutable internal state")

    def predict(self, conditions, genes, knowledge):
        if REFERENCE_ITEM not in knowledge.items:
            raise ValueError("Reference reasoner needs reference_knowledge(), not biological summaries")
        result = []
        strength = knowledge.reliabilities[REFERENCE_ITEM]
        for c in conditions:
            history = [o for o in knowledge.memory.values()
                       if o.condition.context == c.context and o.condition.modality == c.modality]
            votes = np.zeros((len(genes), 3))
            # A fixed non-DE pseudocount; reliability modulates reliance on history.
            votes[:, 1] = 1.0
            for obs in history:
                weight = 2.0 if set(obs.condition.targets) & set(c.targets) else 1.0
                votes[np.arange(len(genes)), obs.labels.astype(int)+1] += strength * weight
            labels = np.argmax(votes, axis=1)-1
            rationale = ("Diagnostic frequency baseline, not a mechanistic rationale. "
                         + str(len(history)) + " revealed same-context conditions were available.")
            result.append(KnowledgePrediction(labels, tuple((REFERENCE_ITEM,) for _ in genes),
                                              tuple(rationale for _ in genes)))
        return result
