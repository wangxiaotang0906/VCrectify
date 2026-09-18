"""Regression checks for adapter contracts and stateful resume boundaries."""
import copy

import numpy as np
import pytest

from vcevo.backbones.reference import (MemoryReasoner, REFERENCE_ITEM,
                                      RidgeBackbone, reference_knowledge)
from vcevo.demo import synthetic_dataset
from vcevo.engine import RunConfig, VCevo
from vcevo.metrics import perturbation_discrimination_score, regression_metrics
from vcevo.types import Condition, KnowledgePrediction, Observation


@pytest.mark.parametrize("bad", [0.8, 257, float("nan")])
def test_prediction_labels_validate_before_integer_conversion(bad):
    with pytest.raises(ValueError):
        KnowledgePrediction(np.array([bad, 0.0]), ((), ()), ("a", "b"))


@pytest.mark.parametrize("bad", [0.8, 257, float("nan")])
def test_observed_labels_validate_before_integer_conversion(bad):
    with pytest.raises(ValueError):
        Observation(Condition("x", "K562", ("x",)), np.zeros(2),
                    np.array([bad, 0.0]), np.ones(2), np.ones((2, 2)))


class StatefulReasoner(MemoryReasoner):
    """Valid stochastic adapter with explicit state and a fresh fold factory."""
    name = "test_stateful_reasoner"

    def __init__(self, seed=123):
        self.rng = np.random.default_rng(seed)

    def fresh(self, seed):
        return StatefulReasoner(seed)

    def predict(self, conditions, genes, knowledge):
        return [KnowledgePrediction(self.rng.choice([-1, 0, 1], len(genes)),
                                    tuple((REFERENCE_ITEM,) for _ in genes),
                                    tuple("seeded test adapter" for _ in genes))
                for condition in conditions]

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


def stateful_engine(path, resume=False):
    data = synthetic_dataset()
    return VCevo(data, RidgeBackbone(data.genes), StatefulReasoner(), reference_knowledge(),
                 RunConfig(budget=4, batch_size=2), path, resume=resume)


def test_stateful_reasoner_resume_matches_uninterrupted(tmp_path):
    complete = stateful_engine(tmp_path / "complete")
    complete.run()
    interrupted = stateful_engine(tmp_path / "resume")
    interrupted.run(max_rounds=1)
    resumed = stateful_engine(tmp_path / "resume", resume=True)
    resumed.run()
    assert resumed.history == complete.history
    assert resumed.pool == complete.pool
    assert resumed.knowledge.reliabilities == complete.knowledge.reliabilities


def test_resume_preserves_observed_memory_order(tmp_path):
    data = synthetic_dataset()
    data.splits["initial"].reverse()
    def construct(resume=False):
        return VCevo(data, RidgeBackbone(data.genes), StatefulReasoner(), reference_knowledge(),
                     RunConfig(budget=4, batch_size=2), tmp_path / "resume", resume=resume)
    interrupted = construct()
    interrupted.run(max_rounds=0)
    before = list(interrupted.knowledge.memory)
    resumed = construct(resume=True)
    assert list(resumed.knowledge.memory) == before


def test_evaluation_does_not_advance_predictive_reasoner_state(tmp_path):
    engine = stateful_engine(tmp_path / "run")
    engine.initialize()
    before = copy.deepcopy(engine.reasoning.state_dict())
    engine.evaluate()
    assert engine.reasoning.state_dict() == before


def test_new_batch_is_not_present_in_its_own_replay():
    calls = []

    class RecordingBackbone(RidgeBackbone):
        def update(self, observations, replay, control_mean, eta):
            calls.append(({o.condition.id for o in observations},
                          {o.condition.id for o in replay}, eta))
            return super().update(observations, replay, control_mean, eta)

    data = synthetic_dataset()
    engine = VCevo(data, RecordingBackbone(data.genes), MemoryReasoner(), reference_knowledge(),
                   RunConfig(budget=2, batch_size=2, eta=0.3, correction="update_all"))
    engine.run()
    assert len(calls) == 1
    new, replay, eta = calls[0]
    assert not new & replay
    assert replay <= set(data.splits["initial"])
    assert eta == 0.3


def test_pds_uses_paper_denominator_and_exact_l1_ranks():
    observed = np.array([[0.0, 0.0], [2.0, 2.0], [6.0, 6.0]])
    assert perturbation_discrimination_score(observed, observed) == 1.0
    # Correct ranks are 3, 1, 3; denominator is N=3, not N-1.
    predicted = observed[::-1]
    assert perturbation_discrimination_score(predicted, observed) == pytest.approx(5 / 9)
    assert regression_metrics(predicted, observed)["pds"] == pytest.approx(5 / 9)


def test_pds_ties_are_average_rank_and_permutation_invariant():
    observed = np.array([[0.0], [2.0]])
    predicted = np.array([[1.0], [1.0]])
    assert perturbation_discrimination_score(predicted, observed) == 0.75
    assert perturbation_discrimination_score(predicted[::-1], observed[::-1]) == 0.75


def test_pds_never_compares_across_cellular_contexts():
    observed = np.array([[0.0], [100.0], [2.0], [102.0]])
    predicted = np.array([[100.0], [0.0], [102.0], [2.0]])
    contexts = ["A", "B", "A", "B"]
    # Each context has one correct rank 1 and one correct rank 2.
    assert perturbation_discrimination_score(predicted, observed, contexts) == 0.75
    assert perturbation_discrimination_score(predicted, observed) != 0.75
    with pytest.raises(ValueError, match="context"):
        perturbation_discrimination_score(predicted, observed, ["A"])
