import numpy as np
import pytest

from vcevo.backbones.reference import RidgeBackbone
from vcevo.metrics import (classification_metrics, deg_subset_metrics,
                           differential_expression_score, regression_metrics)


def test_de_subsets_are_nested_observation_selected_and_report_shortfalls():
    observed = np.array([[4.0, -3.0, 2.0, -1.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    predicted = np.array([[4.0, 3.0, -2.0, -1.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    q = np.array([[.001, .01, .02, .03, .9], [.9, .9, .9, .9, .9]])
    result = deg_subset_metrics(predicted, observed, q, top_ks=(2, 4, 20))
    assert result["dac_de2"] == .5
    assert result["dac_de4"] == .5
    assert result["de2_gene_counts"] == [2, 0]
    assert result["de20_gene_counts"] == [4, 0]
    assert result["de20_defined_conditions"] == 1
    perturbed_predictions = predicted.copy()
    perturbed_predictions[:, 4] = -1e6
    assert deg_subset_metrics(perturbed_predictions, observed, q, top_ks=(2, 4, 20)) == result


def test_de_pcc_zero_variance_and_zero_de_remain_undefined():
    result = deg_subset_metrics([[1, 1], [1, 2]], [[3, 4], [2, 3]], [[.01, .01], [.9, .9]])
    assert result["dac_de20"] == 1
    assert result["pcc_de20"] is None
    assert result["de20_pcc_defined_conditions"] == 0
    empty = deg_subset_metrics([[1, 2]], [[2, 3]], [[.9, .9]])
    assert empty["dac_de40"] is None
    assert empty["de40_defined_conditions"] == 0


def test_condition_macro_rmse_is_not_pooled_rmse():
    result = regression_metrics([[0, 0], [0, 0]], [[0, 0], [3, 4]])
    assert result["rmse"] == pytest.approx(np.sqrt(12.5) / 2)
    assert result["rmse"] != pytest.approx(2.5)
    assert result["pcc"] is None
    assert result["pcc_defined_conditions"] == 0


def test_macro_f1_keeps_all_three_classes_when_one_is_observed():
    result = classification_metrics([[0, 0]], [[0, 0]])
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["accuracy"] == 1
    assert result["per_class"]["down"]["support"] == 0


def cell_groups():
    rng = np.random.default_rng(41)
    controls = rng.uniform(9.5, 10.5, size=(100, 4))
    observed = controls.copy()
    observed[:, 0] *= 4
    observed[:, 1] *= 4
    predicted = controls.copy()
    predicted[:, 0] *= 4
    predicted[:, 1] *= 2
    predicted[:, 2] *= 16
    return predicted, observed, controls


def test_des_uses_actual_rank_sum_and_truncates_by_absolute_log_fold_change():
    predicted, observed, controls = cell_groups()
    result = differential_expression_score(predicted, observed, controls,
        family_scope="full_measured_genes", total_measured_genes=4, expression_scale="linear")
    assert result["observed_de_indices"] == [0, 1]
    assert result["n_predicted_de"] == 3
    assert result["retained_predicted_de_indices"] == [2, 0]
    assert result["des"] == .5
    log_result = differential_expression_score(np.log1p(predicted), np.log1p(observed), np.log1p(controls),
        family_scope="full_measured_genes", total_measured_genes=4, expression_scale="log1p")
    assert log_result["des"] == result["des"]
    assert log_result["retained_predicted_de_indices"] == result["retained_predicted_de_indices"]


def test_des_zero_observed_de_is_undefined_and_subset_scope_mandatory():
    predicted, observed, controls = cell_groups()
    result = differential_expression_score(predicted, controls.copy(), controls,
        family_scope="selected_genes", total_measured_genes=8561, expression_scale="linear")
    assert result["des"] is None
    assert result["n_observed_de"] == 0
    assert result["bh_family_size"] == 4
    assert result["family_scope"] == "selected_genes"
    with pytest.raises(ValueError, match="every measured gene"):
        differential_expression_score(predicted, observed, controls,
            family_scope="full_measured_genes", total_measured_genes=8561)
    with pytest.raises(ValueError, match="cell matrices"):
        differential_expression_score(predicted.mean(axis=0), observed, controls,
            family_scope="selected_genes", total_measured_genes=8561)


def test_des_fewer_predicted_de_uses_observed_set_as_denominator():
    predicted, observed, controls = cell_groups()
    predicted = controls.copy()
    predicted[:, 0] *= 4
    result = differential_expression_score(predicted, observed, controls,
        family_scope="full_measured_genes", total_measured_genes=4, expression_scale="linear")
    assert result["n_predicted_de"] == 1
    assert result["des"] == .5


def test_des_truncation_uses_fold_change_not_absolute_linear_difference():
    rng = np.random.default_rng(12)
    controls = rng.uniform(.99, 1.01, size=(80, 2)) * [100, 1]
    observed = controls.copy()
    observed[:, 0] *= 4
    predicted = controls * [4, 16]
    result = differential_expression_score(predicted, observed, controls,
        family_scope="full_measured_genes", total_measured_genes=2, expression_scale="linear")
    # Gene 0 changes by ~300 but fourfold; gene 1 by ~15 but sixteenfold.
    assert result["retained_predicted_de_indices"] == [1]
    assert result["des"] == 0


def test_ridge_checkpoint_rejects_changed_regularization_or_gene_identity():
    model = RidgeBackbone(("A", "B"), regularization=1.0)
    with pytest.raises(ValueError, match="configuration"):
        RidgeBackbone(("A", "B"), regularization=999).load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="gene axis"):
        RidgeBackbone(("B", "A"), regularization=1.0).load_state_dict(model.state_dict())
    restored = RidgeBackbone(("A", "B"), regularization=1.0)
    restored.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(restored.weights, model.weights)
