import json

import numpy as np
import pytest

from vcevo.artifacts import AuditLog, load_checkpoint, save_checkpoint
from vcevo.backbones.reference import MemoryReasoner, RidgeBackbone, reference_knowledge
from vcevo.demo import synthetic_dataset
from vcevo.engine import OutcomeOracle, RunConfig, VCevo, calibrate_thresholds
from vcevo.policies import Action, Reservoir, acquire, adjudicate, disagreement, errors, update_reliabilities
from vcevo.types import Condition, KnowledgePrediction, Observation


def make_engine(tmp_path=None, config=None, dataset=None, resume=False):
    data = dataset or synthetic_dataset()
    return VCevo(data, RidgeBackbone(data.genes), MemoryReasoner(), reference_knowledge(),
                 config or RunConfig(budget=4, batch_size=2), tmp_path, resume=resume)


def test_four_actions_and_strict_threshold():
    assert adjudicate(1, 1, 1, 1) == Action.NO_UPDATE
    assert adjudicate(2, 1, 1, 1) == Action.CORRECT_M
    assert adjudicate(1, 2, 1, 1) == Action.CORRECT_K
    assert adjudicate(2, 2, 1, 1) == Action.CORRECT_BOTH
    with pytest.raises(ValueError):
        adjudicate(float("nan"), 1, 1, 1)


def test_response_weighted_errors_and_active_zero_class():
    o = Observation(Condition("p", "c", ("p",)), np.array([3., 4.]), np.array([1, -1]),
                    np.array([.01, .01]), np.ones((2, 2)))
    em, ek = errors(np.array([0., 4.]), np.array([0, -1]), o)
    assert em == pytest.approx(np.sqrt(4.5))
    assert ek == pytest.approx(np.sqrt(4.5))
    assert disagreement([[3, 4]], [[0, 0]])[0] == pytest.approx(np.sqrt(12.5))


def test_acquisition_top_half_unique_and_no_candidate_labels():
    ids = ["a", "b", "c", "d", "e", "f"]
    result = acquire(ids, [0, 1, 2, 3, 4, 5], 4, np.random.default_rng(1))
    assert result[:2] == ["f", "e"]
    assert len(set(result)) == 4
    assert acquire(ids, [0]*6, 1, np.random.default_rng(3)) == acquire(ids, [0]*6, 1, np.random.default_rng(3))


def test_reliability_regularized_solution_ema_and_unused():
    data = synthetic_dataset()
    obs = list(data.observations.values())[:2]
    knowledge = reference_knowledge()
    key = next(iter(knowledge.items))
    predictions = [KnowledgePrediction(o.labels, tuple((key, key) for _ in data.genes),
                                       tuple("test" for _ in data.genes)) for o in obs]
    update_reliabilities(knowledge, obs, predictions, initialize=True)
    assert knowledge.reliabilities[key] == 1
    wrong = [KnowledgePrediction(np.where(o.labels == 1, -1, 1), tuple((key,) for _ in data.genes),
                                 tuple("test" for _ in data.genes)) for o in obs]
    update_reliabilities(knowledge, obs, wrong, initialize=True)
    assert knowledge.reliabilities[key] == pytest.approx(1/(1+2*len(data.genes)))
    old = knowledge.reliabilities[key]
    update_reliabilities(knowledge, obs, predictions, eta=.25)
    assert knowledge.reliabilities[key] == pytest.approx(.25+.75*old)
    update_reliabilities(knowledge, [], [], eta=.25)
    assert knowledge.reliabilities[key] == pytest.approx(.25+.75*old)


def test_oracle_disallows_test_and_early_history():
    data = synthetic_dataset()
    oracle = OutcomeOracle(data)
    with pytest.raises(ValueError):
        oracle.reveal(data.splits["test"])
    with pytest.raises(ValueError):
        oracle.history(data.splits["pool"])
    oracle.reveal(data.splits["initial"])
    with pytest.raises(ValueError):
        oracle.reveal(data.splits["initial"])


def test_prediction_arrays_cannot_be_mutated():
    o = next(iter(synthetic_dataset().observations.values()))
    with pytest.raises(ValueError):
        o.delta[0] = 1


def test_lopo_each_fold_excludes_heldout_from_fit_and_memory():
    data = synthetic_dataset()
    seen = []
    class SpyRidge(RidgeBackbone):
        def fresh(self, seed):
            return SpyRidge(self.genes, seed=seed)
        def fit(self, observations, control_mean):
            seen.append({o.condition.id for o in observations})
            super().fit(observations, control_mean)
    class SpyReasoner(MemoryReasoner):
        def fresh(self, seed):
            return self
        def predict(self, conditions, genes, knowledge):
            assert not {c.id for c in conditions} & set(knowledge.memory)
            return super().predict(conditions, genes, knowledge)
    initial = [data.observations[key] for key in data.splits["initial"]]
    _, _, folds = calibrate_thresholds(initial, data.genes, data.control_mean,
                                       SpyRidge(data.genes), SpyReasoner(), reference_knowledge())
    assert len(seen) == len(initial)
    for train, fold in zip(seen, folds):
        assert fold["held_out"] not in train
        assert len(train) == len(initial)-1


def test_reservoir_sampling_seen_count_unique_and_reproducible():
    a, b = Reservoir(3, 42), Reservoir(3, 42)
    a.add([str(i) for i in range(30)])
    b.add([str(i) for i in range(30)])
    assert a.ids == b.ids and a.seen == 30 and len(a.ids) == 3
    with pytest.raises(ValueError):
        a.add(["0"])


def test_prediction_seal_precedes_reveal_and_memory_stays_observed(tmp_path):
    engine = make_engine(tmp_path)
    engine.run()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    for i, event in enumerate(events):
        if event["event"] == "outcomes_revealed_and_adjudicated":
            previous = events[i-1]
            assert previous["event"] == "predictions_sealed"
            assert event["prediction_seal"] == previous["hash"]
            assert not set(previous["selected_ids"]) & set(previous["memory_ids"])
    assert set(engine.knowledge.memory) == engine.oracle.observed
    assert not set(engine.data.splits["test"]) & engine.oracle.observed
    assert not set(engine.data.splits["validation"]) & engine.oracle.observed


def test_unrevealed_outcomes_do_not_change_first_acquisition(tmp_path):
    left = synthetic_dataset()
    right = synthetic_dataset()
    for key in right.splits["pool"] + right.splits["test"]:
        o = right.observations[key]
        right.observations[key] = Observation(o.condition, o.delta*100,
                                             -o.labels, o.qvalues, o.cells*100)
    a = make_engine(tmp_path/"a", dataset=left)
    b = make_engine(tmp_path/"b", dataset=right)
    a.run(max_rounds=1)
    b.run(max_rounds=1)
    def acquisition(path):
        return next(json.loads(line)["selected_ids"] for line in path.read_text().splitlines()
                    if json.loads(line)["event"] == "predictions_sealed")
    assert acquisition(tmp_path/"a/events.jsonl") == acquisition(tmp_path/"b/events.jsonl")


def test_resume_matches_uninterrupted_run(tmp_path):
    full = make_engine(tmp_path/"full")
    full.run()
    interrupted = make_engine(tmp_path/"resume")
    interrupted.run(max_rounds=1)
    resumed = make_engine(tmp_path/"resume", resume=True)
    resumed.run()
    assert resumed.history == full.history
    assert resumed.pool == full.pool
    assert resumed.replay.ids == full.replay.ids
    np.testing.assert_array_equal(resumed.numerical.weights, full.numerical.weights)


def test_no_update_still_admits_observed_memory_and_replay():
    engine = make_engine()
    engine.initialize()
    engine.gamma_m = engine.gamma_k = 1e9
    old_weights = engine.numerical.weights.copy()
    old_rel = dict(engine.knowledge.reliabilities)
    engine.step()
    np.testing.assert_array_equal(engine.numerical.weights, old_weights)
    assert engine.knowledge.reliabilities == old_rel
    assert len(engine.knowledge.memory) == len(engine.data.splits["initial"])+engine.config.batch_size
    assert engine.replay.seen == len(engine.knowledge.memory)


def test_audit_tampering_detected(tmp_path):
    log = AuditLog(tmp_path/"events.jsonl")
    log.append("predicted", labels=[0, 1])
    path = tmp_path/"events.jsonl"
    path.write_text(path.read_text().replace('"predicted"', '"forged"'))
    with pytest.raises(ValueError):
        AuditLog(path, resume=True)


def test_safe_checkpoint_arrays_and_integer_mapping_keys(tmp_path):
    path = tmp_path/"state.npz"
    state = {"weights": np.eye(3), "optimizer": {0: {"step": 2}}, "ids": ("x", "y")}
    save_checkpoint(path, state)
    restored = load_checkpoint(path)
    np.testing.assert_array_equal(restored["weights"], state["weights"])
    assert restored["optimizer"][0]["step"] == 2
    assert restored["ids"] == ("x", "y")


@pytest.mark.parametrize("correction", ["update_all", "random_matched"])
def test_ablation_policies_use_full_loop(correction):
    engine = make_engine(config=RunConfig(budget=4, correction=correction))
    engine.run()
    assert engine.acquired == 4
    assert len(engine.history) == 3
