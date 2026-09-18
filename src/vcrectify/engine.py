"""Predict -> seal -> observe -> adjudicate -> correct.

The numerical/knowledge adapters only receive conditions during inference.
Observed memory is enlarged after prediction and correction, including NoUpdate
conditions. This resolves the manuscript's otherwise implicit memory semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import AuditLog, digest, load_checkpoint, save_checkpoint, write_json
from .metrics import (classification_metrics, regression_metrics, deg_subset_metrics,
                      differential_expression_score)
from .policies import Action, Reservoir, acquire, adjudicate, disagreement, errors, update_reliabilities
from .types import Observation, PreparedDataset


@dataclass
class RunConfig:
    seed: int = 17
    batch_size: int = 2
    budget: int = 8
    replay_capacity: int = 8
    eta: float = 0.5
    threshold_quantile: float = 0.9
    acquisition: str = "disagreement"
    correction: str = "adjudicated"
    evaluate_every: int = 1

    def validate(self):
        if self.batch_size < 1 or self.budget < 0 or self.replay_capacity < 0 or self.evaluate_every < 1:
            raise ValueError("Invalid run sizes")
        if not 0 < self.eta <= 1 or not 0 < self.threshold_quantile < 1:
            raise ValueError("Invalid eta or threshold quantile")
        if self.acquisition not in {"disagreement", "random", "magnitude"}:
            raise ValueError("Unknown acquisition policy")
        if self.correction not in {"adjudicated", "update_all", "random_matched"}:
            raise ValueError("Unknown correction policy")
        return self


class OutcomeOracle:
    """An API barrier against accidental label use, not an adversarial sandbox."""
    def __init__(self, dataset: PreparedDataset):
        self.__observations = dataset.observations
        self.allowed = set(dataset.splits["initial"] + dataset.splits["pool"])
        self.observed = set()

    def conditions(self, ids):
        return [self.__observations[key].condition for key in ids]

    def reveal(self, ids):
        if len(set(ids)) != len(ids) or set(ids) - self.allowed:
            raise ValueError("Only distinct initial/acquisition conditions may be revealed")
        if set(ids) & self.observed:
            raise ValueError("Conditions were already revealed")
        self.observed.update(ids)
        return [self.__observations[key] for key in ids]

    def history(self, ids):
        if set(ids) - self.observed:
            raise ValueError("Attempt to access unrevealed outcomes")
        return [self.__observations[key] for key in ids]


def checked_predict(numerical, reasoning, conditions, genes, control_mean, knowledge):
    deltas = np.asarray(numerical.predict(conditions, control_mean), dtype=float)
    predictions = reasoning.predict(conditions, genes, knowledge)
    if deltas.shape != (len(conditions), len(genes)) or not np.isfinite(deltas).all():
        raise ValueError("Numerical adapter returned invalid shape or non-finite predictions")
    if len(predictions) != len(conditions):
        raise ValueError("Reasoner returned the wrong number of conditions")
    for prediction in predictions:
        if len(prediction.labels) != len(genes):
            raise ValueError("Reasoner returned the wrong gene axis")
        for citations in prediction.citations:
            if set(citations) - set(knowledge.items):
                raise ValueError("Reasoner cited evidence outside the prior knowledge store")
    return deltas.copy(), predictions


def calibrate_thresholds(initial: Sequence[Observation], genes, control_mean, numerical,
                         reasoning, prior, quantile=0.9, seed=0):
    """Exact leave-one-CONDITION-out; no in-sample fallback or test calibration."""
    if len(initial) < 2:
        raise ValueError("LOPO threshold calibration needs at least two initial conditions")
    fold_records = []
    for fold, heldout in enumerate(initial):
        training = [o for o in initial if o.condition.id != heldout.condition.id]
        model = numerical.fresh(seed + fold)
        reasoner = reasoning.fresh(seed + fold)
        knowledge = prior.fresh()
        # Supporting citations are recorded without any training outcomes in memory.
        prior_predictions = reasoner.predict([o.condition for o in training], genes, knowledge)
        update_reliabilities(knowledge, training, prior_predictions, initialize=True)
        knowledge.memory = {o.condition.id: o for o in training}
        model.fit(training, control_mean)
        m, k = checked_predict(model, reasoner, [heldout.condition], genes, control_mean, knowledge)
        em, ek = errors(m[0], k[0].labels, heldout)
        fold_records.append({"held_out": heldout.condition.id,
                             "training_ids": [o.condition.id for o in training],
                             "error_m": em, "error_k": ek})
    gamma_m = float(np.quantile([r["error_m"] for r in fold_records], quantile))
    gamma_k = float(np.quantile([r["error_k"] for r in fold_records], quantile))
    return gamma_m, gamma_k, fold_records


class VCrectify:
    def __init__(self, dataset, numerical, reasoning, knowledge, config=None, output_dir=None,
                 provenance=None, resume=False):
        self.data = dataset.validate()
        self.numerical, self.reasoning = numerical, reasoning
        self.prior = knowledge.fresh()
        self.knowledge = knowledge.fresh()
        self.config = (config or RunConfig()).validate()
        if self.config.budget > len(self.data.splits["pool"]):
            raise ValueError("Acquisition budget exceeds the available candidate pool")
        self.oracle = OutcomeOracle(dataset)
        self.replay = Reservoir(self.config.replay_capacity, self.config.seed + 101)
        self.rng = np.random.default_rng(self.config.seed + 202)
        self.gate_rng = np.random.default_rng(self.config.seed + 303)
        self.pool = list(dataset.splits["pool"])
        self.round = 0
        self.acquired = 0
        self.gamma_m = self.gamma_k = None
        self.history = []
        self.output_dir = Path(output_dir) if output_dir else None
        self.provenance = provenance or {}
        # Dataset labels are fingerprinted by infrastructure, never sent to adapters.
        self.fingerprint = digest({"genes": dataset.genes, "splits": dataset.splits,
                                   "control": dataset.control_mean,
                                   "control_cells_hash": __import__("hashlib").sha256(
                                       dataset.control_cells.tobytes()).hexdigest(),
                                   "outcomes": {k: {"condition": asdict(o.condition),
                                                     "delta": o.delta, "labels": o.labels,
                                                     "qvalues": o.qvalues,
                                                     "cells_hash": __import__("hashlib").sha256(o.cells.tobytes()).hexdigest()}
                                                for k, o in dataset.observations.items()},
                                   "prior": {k: asdict(v) for k, v in knowledge.items.items()}})
        self.log = AuditLog(self.output_dir / "events.jsonl", resume=resume) if self.output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            if not self.output_dir:
                raise ValueError("Resume requires an output directory")
            self.restore(self.output_dir / "checkpoint.npz")
        elif self.output_dir:
            write_json(self.output_dir / "run_manifest.json", {
                "schema_version": 1, "dataset_fingerprint": self.fingerprint,
                "run": asdict(self.config), "numerical_backbone": numerical.name,
                "reasoning_backbone": reasoning.name, "provenance": self.provenance,
                "evaluation_scope": "held-out conditions; metrics never feed acquisition or fitting",
                "memory_policy": "all revealed conditions admitted after correction",
                "calibration": "exact leave-one-perturbation-out on initial conditions only",
                "dataset_metadata": dataset.metadata})

    def _event(self, name, **payload):
        if self.log:
            return self.log.append(name, round=self.round, **payload)
        return digest({"event": name, **payload})

    @staticmethod
    def _records(conditions, deltas, knowledge_predictions):
        return [{"condition": asdict(c), "numerical_delta": deltas[i] if deltas is not None else None,
                 "knowledge_labels": k.labels, "citations": k.citations, "rationales": k.rationales}
                for i, (c, k) in enumerate(zip(conditions, knowledge_predictions))]

    def initialize(self):
        if self.gamma_m is not None:
            raise RuntimeError("Already initialized")
        ids = self.data.splits["initial"]
        conditions = self.oracle.conditions(ids)
        predictions = self.reasoning.predict(conditions, self.data.genes, self.knowledge)
        self._event("initial_prior_predictions_sealed", predictions=self._records(conditions, None, predictions))
        initial = self.oracle.reveal(ids)
        self._event("initial_outcomes_revealed", ids=ids)
        self.gamma_m, self.gamma_k, folds = calibrate_thresholds(
            initial, self.data.genes, self.data.control_mean, self.numerical, self.reasoning,
            self.prior, self.config.threshold_quantile, self.config.seed)
        self.numerical.fit(initial, self.data.control_mean)
        update_reliabilities(self.knowledge, initial, predictions, initialize=True)
        self.knowledge.memory = {o.condition.id: o for o in initial}
        self.replay.add(ids)
        self._event("initialized", gamma_m=self.gamma_m, gamma_k=self.gamma_k, folds=folds,
                    reliabilities=self.knowledge.reliabilities, replay_ids=self.replay.ids)
        self.evaluate()
        self.checkpoint()

    def step(self):
        if self.gamma_m is None:
            raise RuntimeError("Call initialize before acquisition")
        if self.acquired >= self.config.budget:
            return False
        self.round += 1
        conditions = self.oracle.conditions(self.pool)
        m, k = checked_predict(self.numerical, self.reasoning, conditions, self.data.genes,
                               self.data.control_mean, self.knowledge)
        scores = disagreement(m, np.stack([x.labels for x in k]))
        if self.config.acquisition == "magnitude":
            scores = np.sqrt(np.mean(m**2, axis=1))
        selected = acquire(self.pool, scores, min(self.config.batch_size, self.config.budget-self.acquired),
                           self.rng, self.config.acquisition)
        idx = [self.pool.index(key) for key in selected]
        selected_m = m[idx].copy()
        selected_k = [k[i] for i in idx]
        selected_conditions = [conditions[i] for i in idx]
        sealed = self._event("predictions_sealed", candidate_ids=self.pool, scores=scores,
                             selected_ids=selected, memory_ids=sorted(self.knowledge.memory),
                             reliability_digest=digest(self.knowledge.reliabilities),
                             predictions=self._records(selected_conditions, selected_m, selected_k))
        observed = self.oracle.reveal(selected)
        error_pairs = [errors(p, q.labels, o) for p, q, o in zip(selected_m, selected_k, observed)]
        expected = [adjudicate(em, ek, self.gamma_m, self.gamma_k) for em, ek in error_pairs]
        selected_m_idx = [i for i, a in enumerate(expected) if a in {Action.CORRECT_M, Action.CORRECT_BOTH}]
        selected_k_idx = [i for i, a in enumerate(expected) if a in {Action.CORRECT_K, Action.CORRECT_BOTH}]
        if self.config.correction == "update_all":
            selected_m_idx = selected_k_idx = list(range(len(observed)))
        elif self.config.correction == "random_matched":
            selected_m_idx = sorted(self.gate_rng.choice(len(observed), len(selected_m_idx), replace=False).tolist())
            selected_k_idx = sorted(self.gate_rng.choice(len(observed), len(selected_k_idx), replace=False).tolist())
        self._event("outcomes_revealed_and_adjudicated", prediction_seal=sealed,
                    results=[{"id": o.condition.id, "error_m": e[0], "error_k": e[1],
                              "adjudicated_action": a.value, "correct_m": i in selected_m_idx,
                              "correct_k": i in selected_k_idx}
                             for i, (o, e, a) in enumerate(zip(observed, error_pairs, expected))])
        replay = self.oracle.history(self.replay.ids)
        if selected_m_idx:
            self.numerical.update([observed[i] for i in selected_m_idx], replay,
                                  self.data.control_mean, self.config.eta)
        update_reliabilities(self.knowledge, [observed[i] for i in selected_k_idx],
                             [selected_k[i] for i in selected_k_idx], eta=self.config.eta)
        # Memory and reservoir advance only after all frozen predictions were adjudicated.
        self.knowledge.memory.update({o.condition.id: o for o in observed})
        self.replay.add(selected)
        self.pool = [key for key in self.pool if key not in set(selected)]
        self.acquired += len(selected)
        self._event("round_completed", acquired=self.acquired, remaining=len(self.pool),
                    replay_ids=self.replay.ids, reliabilities=self.knowledge.reliabilities,
                    correction_count_m=len(selected_m_idx), correction_count_k=len(selected_k_idx))
        if self.round % self.config.evaluate_every == 0 or self.acquired == self.config.budget:
            self.evaluate()
        self.checkpoint()
        return True

    def evaluate(self):
        ids = self.data.splits["test"]
        if not ids:
            return None
        # This evaluator is the only non-oracle access to test outcomes.
        observations = [self.data.observations[key] for key in ids]
        conditions = [o.condition for o in observations]
        # Evaluation must not consume a stochastic adapter's training/acquisition RNG.
        model_state = deepcopy(self.numerical.state_dict())
        reasoner_state = deepcopy(self.reasoning.state_dict())
        try:
            m, k = checked_predict(self.numerical, self.reasoning, conditions, self.data.genes,
                                   self.data.control_mean, self.knowledge)
            distribution = self._distribution_metrics(conditions, observations)
        finally:
            self.numerical.load_state_dict(model_state)
            self.reasoning.load_state_dict(reasoner_state)
        record = {"round": self.round, "acquired": self.acquired,
                  "numerical": regression_metrics(m, np.stack([o.delta for o in observations]),
                                                   contexts=[c.context for c in conditions]),
                  "knowledge": classification_metrics(np.stack([p.labels for p in k]),
                                                       np.stack([o.labels for o in observations]))}
        record["numerical"].update(deg_subset_metrics(m, np.stack([o.delta for o in observations]),
                                                    np.stack([o.qvalues for o in observations])))
        record["numerical"].update(distribution)
        self.history.append(record)
        self._event("held_out_evaluation", metrics=record)
        if self.output_dir:
            write_json(self.output_dir / "metrics.json", self.history)
            write_json(self.output_dir / "heldout_predictions.json", {
                "genes": self.data.genes, "round": self.round,
                "predictions": self._records(conditions, m, k)})
        return record

    def _distribution_metrics(self, conditions, observations):
        controls = self.data.control_cells
        if not hasattr(self.numerical, "predict_cells"):
            return {"des": None, "des_status": "backbone_has_no_cell_distribution_interface"}
        if len(controls) < 2:
            return {"des": None, "des_status": "matched_control_cell_distribution_unavailable"}
        cells = self.numerical.predict_cells(conditions, controls)
        if len(cells) != len(conditions):
            raise ValueError("predict_cells returned the wrong number of conditions")
        # A free-valued decoder can emit negative log1p expression. Do not
        # silently clip it or invent a distribution just to fill a metric.
        if any(np.any(np.asarray(value) < 0) for value in cells):
            return {"des": None, "des_status": "negative_generated_expression_on_log1p_scale",
                    "des_postprocessing": "none; outputs were not clipped"}
        total = self.data.metadata.get("normalization", {}).get("unique_gene_count", len(self.data.genes))
        scope = "full_measured_genes" if total == len(self.data.genes) else "selected_genes"
        records = [differential_expression_score(predicted, observed.cells, controls,
                    family_scope=scope, total_measured_genes=total, expression_scale="log1p")
                   for predicted, observed in zip(cells, observations)]
        scores = [row["des"] for row in records if row["des"] is not None]
        return {"des": float(np.mean(scores)) if scores else None,
                "des_status": "computed" if scores else "no_observed_de",
                "des_defined_conditions": len(scores), "des_family_scope": scope,
                "des_per_condition": records}

    def checkpoint(self):
        if not self.output_dir:
            return
        state = {"schema_version": 1, "fingerprint": self.fingerprint, "config": asdict(self.config),
                 "numerical_name": self.numerical.name, "reasoning_name": self.reasoning.name,
                 "provenance_digest": digest(self.provenance),
                 "round": self.round, "acquired": self.acquired, "pool": self.pool,
                 "gamma_m": self.gamma_m, "gamma_k": self.gamma_k,
                 "memory_ids": list(self.knowledge.memory), "observed_ids": sorted(self.oracle.observed),
                 "reliabilities": self.knowledge.reliabilities, "replay": self.replay.state_dict(),
                 "rng": self.rng.bit_generator.state, "gate_rng": self.gate_rng.bit_generator.state,
                 "numerical": self.numerical.state_dict(),
                 "reasoning": self.reasoning.state_dict(), "history": self.history,
                 "audit_sequence": self.log.sequence, "audit_hash": self.log.previous}
        save_checkpoint(self.output_dir / "checkpoint.npz", state)

    def restore(self, path):
        state = load_checkpoint(path)
        if state["fingerprint"] != self.fingerprint or state["config"] != asdict(self.config):
            raise ValueError("Dataset/prior or loop configuration differs from checkpoint")
        if state["numerical_name"] != self.numerical.name or state["reasoning_name"] != self.reasoning.name:
            raise ValueError("Backbones differ from checkpoint")
        if state["provenance_digest"] != digest(self.provenance):
            raise ValueError("Backbone configuration/provenance differs from checkpoint")
        if self.log.sequence != state["audit_sequence"] or self.log.previous != state["audit_hash"]:
            raise ValueError("Checkpoint and audit log differ; preserve artifacts and restart a new run")
        self.oracle.observed = set(state["observed_ids"])
        if self.oracle.observed - self.oracle.allowed:
            raise ValueError("Checkpoint contains forbidden observations")
        self.knowledge.memory = {o.condition.id: o for o in self.oracle.history(state["memory_ids"])}
        self.knowledge.reliabilities = state["reliabilities"]
        self.replay.load_state_dict(state["replay"])
        self.rng.bit_generator.state = state["rng"]
        self.gate_rng.bit_generator.state = state["gate_rng"]
        self.numerical.load_state_dict(state["numerical"])
        self.reasoning.load_state_dict(state["reasoning"])
        for key in ["round", "acquired", "pool", "gamma_m", "gamma_k", "history"]:
            setattr(self, key, state[key])

    def run(self, max_rounds=None):
        if self.gamma_m is None:
            self.initialize()
        completed = 0
        while self.acquired < self.config.budget and (max_rounds is None or completed < max_rounds):
            self.step()
            completed += 1
        return self.history
