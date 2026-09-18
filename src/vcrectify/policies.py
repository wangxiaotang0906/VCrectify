"""The paper's equations, without backbone dependencies."""
from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np

from .types import KnowledgePrediction, KnowledgeState, Observation


class Action(str, Enum):
    NO_UPDATE = "NoUpdate"
    CORRECT_M = "CorrectM"
    CORRECT_K = "CorrectK"
    CORRECT_BOTH = "CorrectBoth"


def errors(predicted_delta, predicted_labels, observation: Observation):
    delta = np.asarray(predicted_delta, dtype=float)
    labels = np.asarray(predicted_labels)
    if delta.shape != observation.delta.shape or labels.shape != delta.shape:
        raise ValueError("Prediction/observation gene axes disagree")
    if not np.isfinite(delta).all() or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Invalid prediction")
    numerical = float(np.sqrt(np.mean((delta - observation.delta) ** 2)))
    knowledge = float(np.sqrt(np.mean(observation.delta ** 2 * (labels != observation.labels))))
    return numerical, knowledge


def adjudicate(error_m, error_k, gamma_m, gamma_k):
    if not all(np.isfinite(v) and v >= 0 for v in [error_m, error_k, gamma_m, gamma_k]):
        raise ValueError("Errors and thresholds must be finite and nonnegative")
    bm, bk = error_m > gamma_m, error_k > gamma_k
    return {(False, False): Action.NO_UPDATE, (True, False): Action.CORRECT_M,
            (False, True): Action.CORRECT_K, (True, True): Action.CORRECT_BOTH}[(bm, bk)]


def disagreement(predicted_delta, predicted_labels):
    delta = np.asarray(predicted_delta, dtype=float)
    labels = np.asarray(predicted_labels)
    if delta.ndim != 2 or delta.shape != labels.shape or delta.shape[1] == 0:
        raise ValueError("Acquisition expects aligned [conditions, genes] arrays")
    if not np.isfinite(delta).all() or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Invalid candidate prediction")
    return np.sqrt(np.mean(delta ** 2 * (np.sign(delta) != labels), axis=1))


def acquire(ids, scores, batch_size, rng, strategy="disagreement"):
    if batch_size < 1 or len(set(ids)) != len(ids):
        raise ValueError("A positive batch size and unique candidate IDs are required")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ValueError("Invalid acquisition scores")
    n = min(batch_size, len(ids))
    if strategy not in {"disagreement", "random", "magnitude"}:
        raise ValueError("Unknown acquisition strategy")
    targeted = n // 2 if strategy != "random" else 0
    # IDs break score ties reproducibly, independently of filesystem order.
    ranked = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    selected = ranked[:targeted]
    remaining = [i for i in range(len(ids)) if i not in set(selected)]
    if n > targeted:
        selected += [int(i) for i in rng.choice(remaining, size=n-targeted, replace=False)]
    return [ids[i] for i in selected]


def update_reliabilities(knowledge: KnowledgeState, observations: Sequence[Observation],
                         predictions: Sequence[KnowledgePrediction], eta=0.5, initialize=False):
    if not 0 < eta <= 1 or len(observations) != len(predictions):
        raise ValueError("Invalid reliability update arguments")
    agreements = {}
    for obs, prediction in zip(observations, predictions):
        if len(prediction.labels) != len(obs.labels):
            raise ValueError("Mismatched response genes")
        for gene, citations in enumerate(prediction.citations):
            for item in set(citations):
                if item not in knowledge.items:
                    raise ValueError("Unknown cited evidence item: " + item)
                agreements.setdefault(item, []).append(int(prediction.labels[gene] == obs.labels[gene]))
    if initialize:
        knowledge.reliabilities = {item: 1.0 for item in knowledge.items}
    for item, values in agreements.items():
        if initialize:
            knowledge.reliabilities[item] = (1 + sum(values)) / (1 + len(values))
        else:
            knowledge.reliabilities[item] = eta * float(np.mean(values)) + (1-eta)*knowledge.reliabilities[item]
    return {item: len(values) for item, values in agreements.items()}


class Reservoir:
    """Algorithm R at condition granularity; each condition enters exactly once."""
    def __init__(self, capacity, seed=0):
        if capacity < 0:
            raise ValueError("Replay capacity cannot be negative")
        self.capacity = capacity
        self.ids = []
        self.seen = 0
        self.admitted = set()
        self.rng = np.random.default_rng(seed)

    def add(self, ids):
        for key in ids:
            if key in self.admitted:
                raise ValueError("Condition admitted to reservoir twice: " + key)
            self.admitted.add(key)
            self.seen += 1
            if len(self.ids) < self.capacity:
                self.ids.append(key)
            else:
                slot = int(self.rng.integers(self.seen))
                if slot < self.capacity:
                    self.ids[slot] = key

    def state_dict(self):
        return {"capacity": self.capacity, "ids": self.ids, "seen": self.seen,
                "admitted": sorted(self.admitted), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed on resume")
        self.ids = list(state["ids"])
        self.seen = int(state["seen"])
        self.admitted = set(state["admitted"])
        self.rng.bit_generator.state = state["rng"]
