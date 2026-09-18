"""Small, backbone-independent contracts. Predictions never receive an oracle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple

import numpy as np


def _validated_labels(values):
    """Reject invalid labels before a narrowing integer cast can hide them."""
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.isin(raw, [-1, 0, 1]).all():
        raise ValueError("Labels must be a one-dimensional -1/0/+1 vector")
    labels = np.array(raw, dtype=np.int8, copy=True)
    labels.setflags(write=False)
    return labels


@dataclass(frozen=True)
class Condition:
    id: str
    context: str
    targets: Tuple[str, ...]
    modality: str = "CRISPRi"

    def __post_init__(self):
        if not self.id or not self.context or not self.targets:
            raise ValueError("A condition needs an id, context and intervention targets")


@dataclass(frozen=True)
class Observation:
    condition: Condition
    delta: np.ndarray
    labels: np.ndarray
    qvalues: np.ndarray
    cells: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "labels", _validated_labels(self.labels))
        for name, dtype in [("delta", np.float64),
                            ("qvalues", np.float64), ("cells", np.float32)]:
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        g = len(self.delta)
        if self.delta.shape != (g,) or self.labels.shape != (g,) or self.qvalues.shape != (g,):
            raise ValueError("Observation vectors must use a common one-dimensional gene axis")
        if self.cells.ndim != 2 or self.cells.shape[1] != g or len(self.cells) == 0:
            raise ValueError("Observation needs a nonempty cell matrix on that gene axis")
        if not all(np.isfinite(a).all() for a in [self.delta, self.qvalues, self.cells]):
            raise ValueError("Observation contains non-finite values")
        if not np.isin(self.labels, [-1, 0, 1]).all() or not ((self.qvalues >= 0) & (self.qvalues <= 1)).all():
            raise ValueError("Labels must be -1/0/+1 and q-values in [0, 1]")


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    text: str
    genes: Tuple[str, ...]
    source: str
    kind: str = "summary"
    context: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeState:
    items: Dict[str, EvidenceItem]
    reliabilities: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, Observation] = field(default_factory=dict)

    def __post_init__(self):
        if any(k != v.id or not v.source for k, v in self.items.items()):
            raise ValueError("Evidence items need matching stable IDs and nonempty source attribution")
        if set(self.reliabilities) - set(self.items):
            raise ValueError("Reliability scores refer to unknown items")
        for key in self.items:
            self.reliabilities.setdefault(key, 1.0)
        if any(not np.isfinite(s) or not 0 <= s <= 1 for s in self.reliabilities.values()):
            raise ValueError("Reliabilities must be in [0, 1]")

    def fresh(self) -> "KnowledgeState":
        return KnowledgeState(dict(self.items))


@dataclass(frozen=True)
class KnowledgePrediction:
    labels: np.ndarray
    citations: Tuple[Tuple[str, ...], ...]
    rationales: Tuple[str, ...]

    def __post_init__(self):
        labels = _validated_labels(self.labels)
        if len(self.citations) != len(labels) or len(self.rationales) != len(labels):
            raise ValueError("One citation list and rationale are required per response gene")
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "citations", tuple(tuple(dict.fromkeys(c)) for c in self.citations))
        object.__setattr__(self, "rationales", tuple(self.rationales))


class NumericalBackbone(Protocol):
    name: str

    def fit(self, observations: Sequence[Observation], control_mean: np.ndarray) -> None: ...
    def predict(self, conditions: Sequence[Condition], control_mean: np.ndarray) -> np.ndarray: ...
    def update(self, observations: Sequence[Observation], replay: Sequence[Observation],
               control_mean: np.ndarray, eta: float) -> None: ...
    def fresh(self, seed: int) -> "NumericalBackbone": ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class ReasoningBackbone(Protocol):
    name: str

    def predict(self, conditions: Sequence[Condition], genes: Sequence[str],
                knowledge: KnowledgeState) -> List[KnowledgePrediction]: ...
    def fresh(self, seed: int) -> "ReasoningBackbone": ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass
class PreparedDataset:
    genes: Tuple[str, ...]
    control_mean: np.ndarray
    observations: Dict[str, Observation]
    splits: Dict[str, List[str]]
    metadata: Dict[str, Any]
    control_cells: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    def validate(self):
        if len(set(self.genes)) != len(self.genes) or len(self.genes) < 2:
            raise ValueError("At least two unique genes are required")
        if np.asarray(self.control_mean).shape != (len(self.genes),) or not np.isfinite(self.control_mean).all():
            raise ValueError("Invalid control mean")
        all_ids = [key for values in self.splits.values() for key in values]
        if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(self.observations):
            raise ValueError("Splits must form a disjoint partition of the observations")
        if not {"initial", "pool", "test"}.issubset(self.splits):
            raise ValueError("Expected initial, pool and test splits (validation optional)")
        for key, observation in self.observations.items():
            if key != observation.condition.id or len(observation.delta) != len(self.genes):
                raise ValueError("Misaligned observation")
        return self
