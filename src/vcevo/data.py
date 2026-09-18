"""Auditable, bounded-row preprocessing of Replogle K562 single-cell data.

The default is an explicitly subsampled *real-data smoke experiment*. No
synthetic cells or outcome-dependent perturbation selection are used. Both
legacy/modern AnnData categorical encodings and dense/CSR ``X`` are supported
without requiring anndata or loading the complete source matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np
from scipy.stats import mannwhitneyu

from .types import Condition, Observation, PreparedDataset


SCHEMA_VERSION = 1


def _strings(values):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v)
                       for v in np.asarray(values)])


def read_annotation(group, key: str) -> np.ndarray:
    """Read a plain or old/new categorical AnnData observation/variable field."""
    if key not in group:
        raise ValueError("Missing annotation field {!r}; available: {}".format(key, list(group)))
    field = group[key]
    if isinstance(field, h5py.Group):
        if not {"codes", "categories"}.issubset(field):
            raise ValueError("Unsupported annotation encoding for {}".format(key))
        codes, categories = field["codes"][:], _strings(field["categories"][:])
    elif "__categories" in group and key in group["__categories"]:
        codes, categories = field[:], _strings(group["__categories"][key][:])
    else:
        return _strings(field[:])
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError("Missing/invalid categorical values in {}".format(key))
    return categories[codes.astype(np.int64)]


class H5ExpressionReader:
    """Read small row batches from dense or CSR H5AD expression matrices."""

    def __init__(self, matrix):
        self.matrix = matrix
        if isinstance(matrix, h5py.Dataset):
            self.shape = tuple(matrix.shape)
            self.storage = "dense"
        else:
            encoding = matrix.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            shape = matrix.attrs.get("shape", matrix.attrs.get("h5sparse_shape"))
            sparse_format = matrix.attrs.get("h5sparse_format", "")
            if isinstance(sparse_format, bytes):
                sparse_format = sparse_format.decode()
            if encoding not in ("", "csr_matrix") or sparse_format not in ("", "csr"):
                raise ValueError("Only row-oriented CSR sparse X is supported; convert CSC before preprocessing")
            if shape is None or not {"data", "indices", "indptr"}.issubset(matrix):
                raise ValueError("Unsupported H5AD X encoding")
            self.shape = tuple(int(s) for s in shape)
            self.storage = "csr"
        if len(self.shape) != 2:
            raise ValueError("X must be a two-dimensional cell-by-gene matrix")

    def rows(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or np.any(np.diff(indices) <= 0):
            raise ValueError("Row indices must be nonempty, sorted and unique")
        if indices[0] < 0 or indices[-1] >= self.shape[0]:
            raise ValueError("Row index outside source X")
        if self.storage == "dense":
            return np.asarray(self.matrix[indices, :], dtype=np.float32)
        output = np.zeros((len(indices), self.shape[1]), dtype=np.float32)
        for i, row in enumerate(indices):
            start, stop = self.matrix["indptr"][int(row):int(row) + 2]
            columns = self.matrix["indices"][int(start):int(stop)]
            values = self.matrix["data"][int(start):int(stop)]
            # add.at also handles legal CSR files with duplicate column entries.
            np.add.at(output[i], columns, values)
        return output


def normalize_counts(counts: np.ndarray, target_sum: float = 10000.0) -> np.ndarray:
    """Library-normalize *before* selecting genes, then log1p per cell."""
    if target_sum <= 0 or not np.isfinite(target_sum):
        raise ValueError("target_sum must be a finite positive number")
    values = np.asarray(counts, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative raw counts in X")
    if not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError("X contains non-integer values; supply raw UMI counts, not normalized expression")
    totals = values.sum(axis=1, dtype=np.float64)
    if np.any(totals <= 0):
        raise ValueError("Zero-library cells are unsupported; filter them explicitly before preparing data")
    values = values * (target_sum / totals).astype(np.float32)[:, None]
    return np.log1p(values).astype(np.float32)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment across the supplied gene family."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1 or not len(pvalues) or not np.isfinite(pvalues).all():
        raise ValueError("pvalues must be a nonempty finite vector")
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(pvalues, kind="stable")
    adjusted = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def differential_response(perturbed: np.ndarray, controls: np.ndarray,
                          q_threshold: float = 0.05, gene_batch_size: int = 128):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U), ties corrected, then BH.

    Control and perturbation cells are independent, so a *signed-rank* test is
    inappropriate. The asymptotic test includes continuity and tie corrections;
    all-constant genes have p=1. Tests are chunked on the gene axis.
    """
    if not 0 < q_threshold < 1:
        raise ValueError("q_threshold must lie strictly between 0 and 1")
    if (perturbed.ndim != 2 or controls.ndim != 2 or
            perturbed.shape[1] != controls.shape[1] or
            min(len(perturbed), len(controls)) < 2):
        raise ValueError("At least two cells per group and matching gene axes are required")
    if not np.isfinite(perturbed).all() or not np.isfinite(controls).all():
        raise ValueError("Differential expression received non-finite cells")
    if gene_batch_size < 1:
        raise ValueError("gene_batch_size must be positive")
    pvalues = np.ones(perturbed.shape[1], dtype=np.float64)
    for start in range(0, len(pvalues), gene_batch_size):
        end = min(start + gene_batch_size, len(pvalues))
        with np.errstate(invalid="ignore", divide="ignore"):
            p = mannwhitneyu(perturbed[:, start:end], controls[:, start:end],
                             axis=0, alternative="two-sided", method="asymptotic",
                             use_continuity=True).pvalue
        # The complete-tie case has variance zero and conveys no evidence.
        pvalues[start:end] = np.nan_to_num(p, nan=1.0)
    qvalues = benjamini_hochberg(pvalues)
    delta = perturbed.mean(axis=0, dtype=np.float64) - controls.mean(axis=0, dtype=np.float64)
    labels = (np.sign(delta) * (qvalues < q_threshold)).astype(np.int8)
    return delta, qvalues, labels


def _sample_rows(rows, maximum, rng):
    if maximum is not None and (not isinstance(maximum, int) or maximum < 2):
        raise ValueError("Cell limits must be integers >= 2 or None")
    return np.sort(rng.choice(rows, size=min(maximum, len(rows)), replace=False)
                   if maximum is not None else rows)


def _read_normalized(reader, rows, target_sum, batch_rows, column_groups=None):
    # Only the requested condition/control group is resident on the full gene
    # axis. All source I/O remains bounded by batch_rows.
    width = reader.shape[1] if column_groups is None else len(column_groups)
    output = np.empty((len(rows), width), dtype=np.float32)
    for start in range(0, len(rows), batch_rows):
        stop = min(start + batch_rows, len(rows))
        counts = reader.rows(rows[start:stop])
        if column_groups is not None:
            collapsed = counts[:, [group[0] for group in column_groups]].copy()
            for column, group in enumerate(column_groups):
                if len(group) > 1:
                    collapsed[:, column] += counts[:, group[1:]].sum(axis=1)
            counts = collapsed
        output[start:stop] = normalize_counts(counts, target_sum)
    return output


def _split_size(value: Union[int, float], total: int, name: str) -> int:
    if isinstance(value, float):
        if not 0 <= value < 1:
            raise ValueError("{} fraction must be in [0, 1)".format(name))
        return int(np.floor(value * total))
    if not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a nonnegative integer or fraction".format(name))
    return value


def prepare_k562(path, output_dir=None, *, n_genes: Optional[int] = 64,
                 max_conditions: Optional[int] = 24,
                 max_control_cells: Optional[int] = 128,
                 max_cells_per_condition: Optional[int] = 32,
                 min_cells_per_condition: int = 8,
                 initial_size: Union[int, float] = 8,
                 test_size: Union[int, float] = 4,
                 validation_size: Union[int, float] = 4,
                 seed: int = 17, cell_seed: int = 18,
                 condition_key: str = "gene", control_label: str = "non-targeting",
                 context: str = "K562", target_sum: float = 10000.0,
                 q_threshold: float = 0.05, batch_rows: int = 128,
                 bh_family: str = "all_genes") -> PreparedDataset:
    """Prepare raw Replogle counts using an auditable perturbation-level split.

    ``None`` disables an explicit size cap. Gene selection uses variance of
    normalized sampled controls only, without perturbation expression. Target
    genes are *not* forced onto the response axis; backbones must represent
    intervention targets independently. Split sizes may be counts or fractions;
    the active-acquisition pool contains all remaining conditions.

    BH defaults to all source genes, then projects adjusted values onto the
    selected response axis. ``bh_family='selected_genes'`` is available but is a
    different, explicitly recorded analysis. Cell inference treats cells as
    independent and pools K562 controls; it does not provide donor-level or
    batch-aware causal inference.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if n_genes is not None and (not isinstance(n_genes, int) or n_genes < 2):
        raise ValueError("n_genes must be >= 2 or None")
    if max_conditions is not None and (not isinstance(max_conditions, int) or max_conditions < 3):
        raise ValueError("max_conditions must be >= 3 or None")
    if min_cells_per_condition < 2 or batch_rows < 1:
        raise ValueError("Require min_cells_per_condition >= 2 and batch_rows >= 1")
    if bh_family not in {"all_genes", "selected_genes"}:
        raise ValueError("bh_family must be all_genes or selected_genes")
    condition_rng = np.random.default_rng(seed)
    cell_rng = np.random.default_rng(cell_seed)
    split_rng = np.random.default_rng(np.random.SeedSequence([seed, 271828]))
    observations: Dict[str, Observation] = {}
    with h5py.File(path, "r") as source:
        reader = H5ExpressionReader(source["X"])
        annotations = read_annotation(source["obs"], condition_key)
        gene_key = "gene_name" if "gene_name" in source["var"] else source["var"].attrs.get("_index", "_index")
        if isinstance(gene_key, bytes):
            gene_key = gene_key.decode()
        all_genes = read_annotation(source["var"], gene_key)
        id_key = "gene_id" if "gene_id" in source["var"] else gene_key
        all_gene_ids = read_annotation(source["var"], id_key)
        if len(annotations) != reader.shape[0] or len(all_genes) != reader.shape[1]:
            raise ValueError("Annotation axes do not match X")
        if any(not gene.strip() for gene in all_genes):
            raise ValueError("Empty gene symbols are unsupported")
        symbol_columns = {}
        for index, symbol in enumerate(all_genes):
            symbol_columns.setdefault(str(symbol), []).append(index)
        column_groups = list(symbol_columns.values())
        merged_gene_ids = [[str(all_gene_ids[i]) for i in group] for group in column_groups]
        duplicate_mapping = {name: {"source_columns": columns,
                             "gene_ids": [str(all_gene_ids[i]) for i in columns]}
                             for name, columns in symbol_columns.items() if len(columns) > 1}
        all_genes = np.asarray(list(symbol_columns))
        control_rows_all = np.flatnonzero(annotations == control_label)
        if len(control_rows_all) < 2:
            raise ValueError("Fewer than two controls match {!r}".format(control_label))
        control_rows = _sample_rows(control_rows_all, max_control_cells, cell_rng)
        controls = _read_normalized(reader, control_rows, target_sum, batch_rows, column_groups)
        variance = controls.var(axis=0, dtype=np.float64)
        # Stable ordering resolves variance ties with the original gene axis.
        selected = np.argsort(-variance, kind="stable")[:n_genes]
        selected = np.sort(selected)
        genes = tuple(str(g) for g in all_genes[selected])
        candidate_names, candidate_counts = np.unique(annotations, return_counts=True)
        candidates = [str(name) for name, count in zip(candidate_names, candidate_counts)
                      if name != control_label and count >= min_cells_per_condition]
        if len(candidates) < 3:
            raise ValueError("Need at least three eligible perturbation conditions")
        if max_conditions is not None and len(candidates) > max_conditions:
            candidates = sorted(condition_rng.choice(candidates, max_conditions, replace=False).tolist())
        n_initial = _split_size(initial_size, len(candidates), "initial_size")
        n_test = _split_size(test_size, len(candidates), "test_size")
        n_validation = _split_size(validation_size, len(candidates), "validation_size")
        if n_initial < 1 or n_test < 1 or n_initial + n_test + n_validation >= len(candidates):
            raise ValueError("Require nonempty initial, pool and test splits; lower split sizes for this condition count")
        randomized = split_rng.permutation(candidates).tolist()
        splits = {
            "initial": randomized[:n_initial],
            "pool": randomized[n_initial:len(candidates) - n_test - n_validation],
            "test": randomized[len(candidates) - n_test - n_validation:len(candidates) - n_validation if n_validation else None],
            "validation": randomized[-n_validation:] if n_validation else [],
        }
        row_manifest = {}
        counts_manifest = {}
        for name in sorted(candidates):
            source_rows = np.flatnonzero(annotations == name)
            rows = _sample_rows(source_rows, max_cells_per_condition, cell_rng)
            perturbed = _read_normalized(reader, rows, target_sum, batch_rows, column_groups)
            if bh_family == "all_genes":
                delta, qvalues, labels = differential_response(perturbed, controls, q_threshold)
                delta, qvalues, labels = delta[selected], qvalues[selected], labels[selected]
            else:
                delta, qvalues, labels = differential_response(perturbed[:, selected], controls[:, selected], q_threshold)
            condition = Condition(id=name, context=context, targets=(name,), modality="CRISPRi")
            observations[name] = Observation(condition, delta, labels, qvalues, perturbed[:, selected])
            row_manifest[name] = rows.tolist()
            counts_manifest[name] = {"available": int(len(source_rows)), "retained": int(len(rows))}
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "Replogle K562 essential single-cell CRISPRi",
            "source_file": str(path.resolve()),
            "source_file_bytes": path.stat().st_size,
            "source_shape": list(reader.shape), "source_storage": reader.storage,
            "source_raw_counts": True,
            "experiment_kind": "real_data_subsample" if (
                len(genes) < len(all_genes) or len(control_rows) < len(control_rows_all) or
                len(candidates) < sum(name != control_label and count >= min_cells_per_condition
                                      for name, count in zip(candidate_names, candidate_counts)) or
                any(item["retained"] < item["available"] for item in counts_manifest.values())) else "real_data_full",
            "normalization": {"method": "library_size_then_log1p", "target_sum": target_sum,
                              "library_size_source_feature_count": int(reader.shape[1]),
                              "unique_gene_count": int(len(all_genes)),
                              "duplicate_symbols": "sum_raw_counts_before_normalization",
                              "merged_symbols": duplicate_mapping},
            "gene_selection": {"method": "normalized_control_variance", "perturbation_expression_used": False,
                               "force_targets": False, "source_indices": [column_groups[i][0] for i in selected],
                               "source_column_groups": [column_groups[i] for i in selected],
                               "gene_ids": [merged_gene_ids[i] for i in selected]},
            "de_test": {"method": "mannwhitneyu_two_sided_asymptotic", "tie_correction": True,
                        "continuity_correction": True, "adjustment": "benjamini_hochberg",
                        "family": bh_family, "family_size": len(all_genes) if bh_family == "all_genes" else len(genes),
                        "q_threshold": q_threshold, "threshold_operator": "<",
                        "control_matching": "pooled_cellular_context", "context": context,
                        "inference_unit": "cell", "batch_adjustment": False},
            "sampling": {"seed": seed, "cell_seed": cell_seed,
                         "max_conditions": max_conditions, "max_control_cells": max_control_cells,
                         "max_cells_per_condition": max_cells_per_condition,
                         "min_cells_per_condition": min_cells_per_condition,
                         "selection": "uniform_random_without_replacement"},
            "control_label": control_label, "condition_key": condition_key,
            "control_count_available": int(len(control_rows_all)),
            "control_count_retained": int(len(control_rows)),
            "control_source_rows": control_rows.tolist(),
            "condition_source_rows": row_manifest, "condition_counts": counts_manifest,
            "split_unit": "perturbation_condition", "split_expression_used": False,
        }
        data = PreparedDataset(genes=genes,
                               control_mean=controls[:, selected].mean(axis=0, dtype=np.float64),
                               observations=observations, splits=splits, metadata=metadata,
                               control_cells=controls[:, selected].copy()).validate()
    if output_dir is not None:
        save_prepared(data, output_dir)
    return data


def save_prepared(dataset: PreparedDataset, path) -> Path:
    """Write a JSON manifest and a pickle-free compressed numerical artifact."""
    dataset.validate()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"control_mean": dataset.control_mean, "control_cells": dataset.control_cells}
    conditions = []
    for i, (key, observation) in enumerate(sorted(dataset.observations.items())):
        prefix = "condition_{}".format(i)
        conditions.append({"id": key, "context": observation.condition.context,
                           "targets": list(observation.condition.targets),
                           "modality": observation.condition.modality, "array_prefix": prefix})
        for field in ("delta", "labels", "qvalues", "cells"):
            arrays[prefix + "_" + field] = getattr(observation, field)
    manifest = {"schema_version": SCHEMA_VERSION, "genes": list(dataset.genes),
                "conditions": conditions, "splits": dataset.splits, "metadata": dataset.metadata}
    # Files use bounded, explicit names and never pickle Python objects.
    np.savez_compressed(path / "dataset.npz", **arrays)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_prepared(path) -> PreparedDataset:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared-dataset schema version")
    observations = {}
    with np.load(path / "dataset.npz", allow_pickle=False) as arrays:
        for item in manifest["conditions"]:
            prefix = item["array_prefix"]
            condition = Condition(item["id"], item["context"], tuple(item["targets"]), item["modality"])
            observations[item["id"]] = Observation(condition, *(arrays[prefix + "_" + field]
                for field in ("delta", "labels", "qvalues", "cells")))
        dataset = PreparedDataset(tuple(manifest["genes"]), arrays["control_mean"].copy(), observations,
                                  manifest["splits"], manifest["metadata"], arrays["control_cells"].copy())
    return dataset.validate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Raw Replogle K562 H5AD")
    parser.add_argument("output", type=Path, help="Prepared dataset directory")
    parser.add_argument("--genes", type=int, default=64)
    parser.add_argument("--conditions", type=int, default=24)
    parser.add_argument("--control-cells", type=int, default=128)
    parser.add_argument("--cells-per-condition", type=int, default=32)
    parser.add_argument("--initial", type=int, default=8)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--validation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cell-seed", type=int, default=18)
    args = parser.parse_args(argv)
    data = prepare_k562(args.source, args.output, n_genes=args.genes, max_conditions=args.conditions,
                        max_control_cells=args.control_cells, max_cells_per_condition=args.cells_per_condition,
                        initial_size=args.initial, test_size=args.test, validation_size=args.validation,
                        seed=args.seed, cell_seed=args.cell_seed)
    print(json.dumps({"output": str(args.output.resolve()), "genes": len(data.genes),
                      "conditions": len(data.observations), "splits": {k: len(v) for k, v in data.splits.items()},
                      "experiment_kind": data.metadata["experiment_kind"]}, indent=2))


if __name__ == "__main__":
    main()
