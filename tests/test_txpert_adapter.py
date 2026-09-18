"""Integration tests execute the actual upstream TxPert, never a fake model.

They skip when optional source/dependencies are absent. Core installation remains
lightweight. Test graph and cell values are deliberately synthetic fixtures.
"""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from vcevo.backbones.txpert import TxPertBackbone
from vcevo.types import Condition, Observation


@pytest.fixture
def model_args(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("VCEVO_TXPERT_SOURCE", root / "third_party/TxPert-main"))
    deps = root / ".runtime/txpert_deps"
    if deps.is_dir() and str(deps) not in sys.path:
        sys.path.insert(0, str(deps))
    if not (source / "gspp/models/txpert.py").is_file():
        pytest.skip("Official TxPert source not installed")
    if any(importlib.util.find_spec(name) is None for name in ["torch", "torch_geometric", "torch_scatter", "omegaconf", "timm"]):
        pytest.skip("Optional TxPert dependencies not installed")
    graph = tmp_path / "fixture_prior.csv"
    graph.write_text("source,target,importance\nA,B,1\nB,A,1\nB,C,1\nC,D,1\nD,A,1\n", encoding="utf-8")
    controls = np.array([[0.8, 1., 1.2, 1.], [1.2, 1., 0.8, 1.]], dtype=np.float32)
    return dict(genes=("A", "B", "C", "D"), targets=("A", "B", "C", "D"),
                graph_path=str(graph), source_dir=str(source), control_cells=controls,
                fit_epochs=6, update_epochs=2, hidden_dim=16, latent_dim=8,
                graph_hidden_dim=8, graph_layers=1, batch_size=2, cells_per_condition=2,
                lr=.01, dropout=0., device="cpu")


def records():
    result = []
    for i, name in enumerate(["A", "B", "C"]):
        delta = np.array([.2, -.1, .15, .05]) * (i + 1)
        cells = np.tile(1. + delta, (2, 1))
        result.append(Observation(Condition(name, "K562", (name,)), delta,
                                  np.sign(delta), np.zeros(4), cells))
    return result


def test_missing_graph_fails_without_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="prior graph"):
        TxPertBackbone(("A", "B"), ("A",), graph_path=tmp_path / "missing.csv")


def test_official_model_fit_prediction_and_cell_distribution(model_args):
    model = TxPertBackbone(**model_args)
    observations = records()
    model.fit(observations[:2], np.ones(4))
    assert type(model.model).__module__ == "gspp.models.txpert"
    assert model.history[-1] < model.history[0]
    conditions = [o.condition for o in observations]
    per_cell = model.predict_cells(conditions, model_args["control_cells"])
    prediction = model.predict(conditions, np.ones(4))
    assert prediction.shape == (3, 4)
    for i, value in enumerate(per_cell):
        assert value.shape == (2, 4)
        np.testing.assert_allclose(prediction[i], (value - model_args["control_cells"]).mean(0), atol=1e-7)
    assert np.isfinite(prediction).all()
    with pytest.raises(ValueError, match="Undeclared"):
        model.predict([Condition("X", "K562", ("X",))], np.ones(4))


def test_checkpoint_reproduces_future_training_and_empty_update(model_args):
    model = TxPertBackbone(**dict(model_args, dropout=.2))
    obs = records()
    model.fit(obs[:2], np.ones(4))
    restored = model.fresh(99)
    assert not restored.fitted
    restored.load_state_dict(model.state_dict())
    before = restored.predict([o.condition for o in obs], np.ones(4))
    restored.update([], obs, np.ones(4), .5)
    np.testing.assert_array_equal(before, restored.predict([o.condition for o in obs], np.ones(4)))
    for item in [model, restored]:
        item.update(obs[2:], obs[:2], np.ones(4), .3)
    np.testing.assert_array_equal(model.predict([o.condition for o in obs], np.ones(4)),
                                  restored.predict([o.condition for o in obs], np.ones(4)))


def test_checkpoint_rejects_changed_training_semantics(model_args):
    model = TxPertBackbone(**model_args)
    changed = TxPertBackbone(**dict(model_args, cells_per_condition=1))
    with pytest.raises(ValueError, match="configuration"):
        changed.load_state_dict(model.state_dict())


def test_eta_one_ignores_replay_and_gene_axis_is_validated(model_args):
    model = TxPertBackbone(**model_args)
    obs = records()
    model.fit(obs[:2], np.ones(4))
    copied = model.fresh(2)
    copied.load_state_dict(model.state_dict())
    model.update(obs[2:], obs[:2], np.ones(4), 1.)
    copied.update(obs[2:], [], np.ones(4), 1.)
    np.testing.assert_array_equal(model.predict([obs[2].condition], np.ones(4)),
                                  copied.predict([obs[2].condition], np.ones(4)))
    with pytest.raises(ValueError, match="does not match"):
        model.predict([obs[0].condition], np.zeros(4))


def test_replay_weight_does_not_depend_on_number_of_conditions(model_args):
    # Identical control cells and complete target-cell sampling eliminate Monte
    # Carlo differences; duplicated replay must retain the same objective mass.
    args = dict(model_args, control_cells=np.ones((2, 4), dtype=np.float32))
    model = TxPertBackbone(**args)
    obs = records()
    model.fit(obs[:2], np.ones(4))
    copied = model.fresh(2)
    copied.load_state_dict(model.state_dict())
    model.update(obs[2:], obs[:2], np.ones(4), .2)
    copied.update(obs[2:], obs[:2] * 3, np.ones(4), .2)
    np.testing.assert_allclose(model.predict([obs[2].condition], np.ones(4)),
                               copied.predict([obs[2].condition], np.ones(4)), atol=2e-6)


def test_training_preserves_other_backbones_torch_rng(model_args):
    import torch
    model = TxPertBackbone(**dict(model_args, dropout=.2))
    torch.random.default_generator.manual_seed(719)
    before = torch.random.get_rng_state().clone()
    model.fit(records()[:2], np.ones(4))
    assert torch.equal(before, torch.random.get_rng_state())
