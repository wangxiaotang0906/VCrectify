"""Minimal external factory example: a diagnostic global mean-effect predictor.

Set PYTHONPATH to include examples, then use numerical.name:
custom_backbone:make_numerical. This intentionally simple predictor has no
perturbation-specific biological knowledge and is not a scientific baseline.
"""
import numpy as np


class MeanEffect:
    name = "example_mean_effect"

    def __init__(self, genes):
        self.genes = tuple(genes)
        self.effect = np.zeros(len(genes))

    def fit(self, observations, control_mean):
        self.effect = np.mean([o.delta for o in observations], axis=0)

    def predict(self, conditions, control_mean):
        return np.tile(self.effect, (len(conditions), 1))

    def update(self, observations, replay, control_mean, eta):
        if observations:
            current = np.mean([o.delta for o in observations], axis=0)
            self.effect = eta*current + (1-eta)*np.mean([o.delta for o in replay], axis=0) if replay else current

    def fresh(self, seed):
        return MeanEffect(self.genes)

    def state_dict(self):
        return {"genes": self.genes, "effect": self.effect.copy()}

    def load_state_dict(self, state):
        if tuple(state["genes"]) != self.genes:
            raise ValueError("Mismatched response-gene axis")
        self.effect = np.asarray(state["effect"]).copy()


def make_numerical(*, genes, targets, control_cells, **options):
    return MeanEffect(genes)
