"""Small H5AD fixtures verify data semantics without redistributing raw cells."""
import json

import h5py
import numpy as np
import pytest
from scipy import sparse

from vcrectify.data import (H5ExpressionReader, benjamini_hochberg, differential_response,
                        load_prepared, normalize_counts, prepare_k562)


def make_h5ad(path, storage="dense", categorical="legacy", perturbation_scale=1):
    rng = np.random.default_rng(40)
    names = np.array(["non-targeting", "A", "B", "C", "D", "E", "F", "G", "H"])
    labels = np.repeat(np.arange(len(names)), 12)
    x = rng.poisson(np.array([1, 2, 8, 0.2, 4, 10]), size=(len(labels), 6)).astype(np.float32)
    x[labels > 0, 0] *= perturbation_scale
    with h5py.File(path, "w") as f:
        if storage == "dense":
            f.create_dataset("X", data=x)
        else:
            matrix = sparse.csr_matrix(x)
            group = f.create_group("X")
            group.attrs["encoding-type"] = "csr_matrix"
            group.attrs["shape"] = x.shape
            for key in ("data", "indices", "indptr"):
                group.create_dataset(key, data=getattr(matrix, key))
        obs = f.create_group("obs")
        var = f.create_group("var")
        genes = np.array(["G0", "G1", "G2", "G3", "G4", "G5"], dtype="S")
        if categorical == "legacy":
            obs.create_dataset("gene", data=labels)
            obs.create_group("__categories").create_dataset("gene", data=names.astype("S"))
            var.create_dataset("gene_name", data=np.arange(6))
            var.create_group("__categories").create_dataset("gene_name", data=genes)
        else:
            group = obs.create_group("gene")
            group.create_dataset("codes", data=labels)
            group.create_dataset("categories", data=names.astype("S"))
            var.create_dataset("gene_name", data=genes)
        var.create_dataset("gene_id", data=np.array(["ID" + str(i) for i in range(6)], dtype="S"))
    return x


def prepare_fixture(path, output=None, **kwargs):
    return prepare_k562(path, output, n_genes=4, max_conditions=8,
                        max_control_cells=10, max_cells_per_condition=8,
                        initial_size=2, test_size=2, validation_size=1,
                        batch_rows=3, **kwargs)


@pytest.mark.parametrize("storage,categorical", [("dense", "legacy"), ("csr", "modern")])
def test_prepare_roundtrip_and_real_statistics(tmp_path, storage, categorical):
    source = tmp_path / "raw.h5ad"
    raw = make_h5ad(source, storage, categorical)
    output = tmp_path / "prepared"
    data = prepare_fixture(source, output)
    loaded = load_prepared(output)
    assert loaded.genes == data.genes
    assert {k: len(v) for k, v in loaded.splits.items()} == {"initial": 2, "pool": 3, "test": 2, "validation": 1}
    assert loaded.metadata["de_test"]["family_size"] == 6
    all_ids = sum(loaded.splits.values(), [])
    assert len(set(all_ids)) == len(all_ids) == 8
    selected = data.metadata["gene_selection"]["source_indices"]
    c = normalize_counts(raw[data.metadata["control_source_rows"]])
    for key, observation in loaded.observations.items():
        np.testing.assert_array_equal(observation.cells, data.observations[key].cells)
        p = normalize_counts(raw[data.metadata["condition_source_rows"][key]])
        delta, qvalues, labels = differential_response(p, c)
        np.testing.assert_allclose(observation.delta, delta[selected])
        np.testing.assert_allclose(observation.qvalues, qvalues[selected])
        np.testing.assert_array_equal(observation.labels, labels[selected])
        np.testing.assert_array_equal(observation.cells, p[:, selected])
    assert loaded.metadata["experiment_kind"] == "real_data_subsample"
    assert json.loads((output / "manifest.json").read_text())["schema_version"] == 1


def test_gene_selection_and_splits_do_not_see_perturbation_expression(tmp_path):
    first, second = tmp_path / "first.h5ad", tmp_path / "second.h5ad"
    make_h5ad(first)
    make_h5ad(second, perturbation_scale=1000)
    a, b = prepare_fixture(first), prepare_fixture(second)
    assert a.genes == b.genes
    assert a.splits == b.splits
    np.testing.assert_array_equal(a.control_cells, b.control_cells)
    assert any(not np.allclose(a.observations[k].delta, b.observations[k].delta) for k in a.observations)


def test_normalization_precedes_gene_filtering():
    counts = np.array([[1, 9], [10, 10]], dtype=np.float32)
    expected = np.log1p(np.array([[1000, 9000], [5000, 5000]], dtype=np.float32))
    np.testing.assert_allclose(normalize_counts(counts), expected)
    with pytest.raises(ValueError, match="Zero-library"):
        normalize_counts(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_counts(np.array([[-1, 2]]))
    with pytest.raises(ValueError, match="raw UMI"):
        normalize_counts(np.array([[0.3, 1.7]]))


def test_bh_and_direction_labels_handle_ties():
    np.testing.assert_allclose(benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.8])),
                               [0.04, 0.0533333333333, 0.0533333333333, 0.8])
    controls = np.tile([0.0, 1.0, 3.0], (20, 1))
    perturbed = np.tile([2.0, 1.0, 0.0], (20, 1))
    delta, qvalues, labels = differential_response(perturbed, controls)
    np.testing.assert_array_equal(labels, [1, 0, -1])
    assert qvalues[1] == 1
    np.testing.assert_array_equal(delta, [2, 0, -3])


def test_seed_reproducibility_and_condition_level_holdout(tmp_path):
    source = tmp_path / "raw.h5ad"
    make_h5ad(source)
    a, b = prepare_fixture(source), prepare_fixture(source)
    assert a.splits == b.splits
    assert a.metadata["condition_source_rows"] == b.metadata["condition_source_rows"]
    source_rows = [r for rows in a.metadata["condition_source_rows"].values() for r in rows]
    assert len(source_rows) == len(set(source_rows))
    assert not set(source_rows) & set(a.metadata["control_source_rows"])


def test_invalid_or_absent_controls_fail_explicitly(tmp_path):
    source = tmp_path / "raw.h5ad"
    make_h5ad(source)
    with pytest.raises(ValueError, match="controls"):
        prepare_fixture(source, control_label="missing")
    with h5py.File(source, "a") as f:
        f["obs/gene"][0] = -1
    with pytest.raises(ValueError, match="categorical"):
        prepare_fixture(source)


def test_sparse_reader_matches_dense_selected_rows(tmp_path):
    source = tmp_path / "raw.h5ad"
    raw = make_h5ad(source, storage="csr")
    with h5py.File(source, "r") as f:
        reader = H5ExpressionReader(f["X"])
        np.testing.assert_array_equal(reader.rows(np.array([0, 5, 7, 80])), raw[[0, 5, 7, 80]])
        with pytest.raises(ValueError, match="sorted"):
            reader.rows(np.array([7, 2]))


def test_duplicate_symbols_sum_counts_before_normalizing(tmp_path):
    source = tmp_path / "duplicate.h5ad"
    raw = make_h5ad(source, categorical="modern")
    with h5py.File(source, "a") as f:
        f["var/gene_name"][1] = b"G0"
    data = prepare_fixture(source)
    assert data.metadata["de_test"]["family_size"] == 5
    assert data.metadata["normalization"]["merged_symbols"]["G0"]["source_columns"] == [0, 1]
    counts = raw[:, [0, 2, 3, 4, 5]].copy()
    counts[:, 0] += raw[:, 1]
    normalized = normalize_counts(counts[data.metadata["control_source_rows"]])
    collapsed_genes = ["G0", "G2", "G3", "G4", "G5"]
    selected = [collapsed_genes.index(gene) for gene in data.genes]
    np.testing.assert_allclose(data.control_cells, normalized[:, selected])
