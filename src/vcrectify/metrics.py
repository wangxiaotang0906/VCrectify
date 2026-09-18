"""Condition-macro metrics. Undefined correlations are reported, never coerced."""
import numpy as np


def perturbation_discrimination_score(predicted, observed, contexts=None):
    """Paper PDS: mean of ``1 - (rank - 1) / N_context`` over conditions.

    Rank the correct observed mean-effect vector among all evaluated conditions
    in its cellular context, using ascending L1 distance. Exact distance ties
    receive their average rank, so input ordering never breaks indistinguishable
    outcomes. The manuscript leaves tie handling unspecified; this convention
    is explicit. Only mean effects are required, not cell-level distributions.

    PDS depends on the evaluated gene and condition sets. A selected-gene smoke
    run is valid on its declared subset but is not a full-transcriptome score.
    A context with a single condition scores 1 by the paper formula and carries
    no discrimination information.
    """
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if (predicted.shape != observed.shape or predicted.ndim != 2 or
            len(observed) == 0 or predicted.shape[1] == 0):
        raise ValueError("Expected nonempty, aligned condition-by-gene arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("PDS requires finite inputs")
    contexts = ["__single_context__"] * len(observed) if contexts is None else list(contexts)
    if len(contexts) != len(observed) or any(not isinstance(c, str) or not c for c in contexts):
        raise ValueError("PDS needs one nonempty cellular-context string per condition")
    scores = []
    for context in dict.fromkeys(contexts):
        group = np.flatnonzero(np.asarray(contexts) == context)
        references = observed[group]
        for index, prediction in enumerate(predicted[group]):
            distances = np.abs(references - prediction).sum(axis=1)
            correct = distances[index]
            rank = 1 + np.count_nonzero(distances < correct) + (np.count_nonzero(distances == correct) - 1) / 2
            scores.append(1 - (rank - 1) / len(group))
    return float(np.mean(scores))


def regression_metrics(predicted, observed, contexts=None):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if (predicted.shape != observed.shape or predicted.ndim != 2 or
            len(observed) == 0 or predicted.shape[1] == 0):
        raise ValueError("Expected nonempty, aligned condition-by-gene arrays")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("Metrics require finite inputs")
    pearsons = []
    for p, y in zip(predicted, observed):
        if np.std(p) == 0 or np.std(y) == 0:
            pearsons.append(np.nan)
        else:
            pearsons.append(float(np.corrcoef(p, y)[0, 1]))
    return {"rmse": float(np.sqrt(np.mean((predicted-observed)**2, axis=1)).mean()),
            "mae": float(np.abs(predicted-observed).mean(axis=1).mean()),
            "dac": float((np.sign(predicted) == np.sign(observed)).mean(axis=1).mean()),
            "pds": perturbation_discrimination_score(predicted, observed, contexts),
            "pcc": float(np.nanmean(pearsons)) if np.isfinite(pearsons).any() else None,
            "pcc_defined_conditions": int(np.isfinite(pearsons).sum()),
            "n_conditions": len(observed)}


def classification_metrics(predicted, observed):
    predicted, observed = np.asarray(predicted).ravel(), np.asarray(observed).ravel()
    if predicted.shape != observed.shape or len(predicted) == 0:
        raise ValueError("Expected aligned labels")
    per_class = {}
    f1s = []
    for label, name in [(-1, "down"), (0, "non_de"), (1, "up")]:
        tp = int(((predicted == label) & (observed == label)).sum())
        fp = int(((predicted == label) & (observed != label)).sum())
        fn = int(((predicted != label) & (observed == label)).sum())
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        f1s.append(f1)
        per_class[name] = {"f1": f1, "support": tp+fn,
                           "precision": tp/(tp+fp) if tp+fp else 0.0,
                           "recall": tp/(tp+fn) if tp+fn else 0.0}
    return {"macro_f1": float(np.mean(f1s)), "accuracy": float((predicted == observed).mean()),
            "per_class": per_class, "n_queries": len(observed)}


def deg_subset_metrics(predicted, observed, observed_qvalues, *, top_ks=(20, 40), q_threshold=0.05):
    """DAC/PCC on nested top-k *observed significant* response-gene subsets.

    The appendix specifies strongest observed DE responses but does not resolve
    ranking ties or insufficient DE genes. This explicit implementation ranks
    observed q<=threshold genes by descending absolute mean log-expression
    change, breaking ties with the supplied gene-axis position. Fewer than k DE
    genes means all available DE genes; a zero-DE condition is undefined and
    excluded, with all counts reported. Prediction values never select genes.
    """
    predicted, observed, qvalues = [np.asarray(value, dtype=float)
                                    for value in (predicted, observed, observed_qvalues)]
    if (predicted.shape != observed.shape or predicted.shape != qvalues.shape or
            predicted.ndim != 2 or not len(predicted) or not predicted.shape[1]):
        raise ValueError("Expected matching nonempty condition-by-gene prediction, outcome and q-value arrays")
    if not all(np.isfinite(value).all() for value in (predicted, observed, qvalues)):
        raise ValueError("DE subset metrics require finite inputs")
    if not 0 < q_threshold < 1 or np.any((qvalues < 0) | (qvalues > 1)):
        raise ValueError("Invalid q-values or significance threshold")
    top_ks = tuple(top_ks)
    if not top_ks or len(set(top_ks)) != len(top_ks) or any(type(k) is not int or k < 1 for k in top_ks):
        raise ValueError("top_ks must contain distinct positive integers")
    ranks = []
    for effects, q in zip(observed, qvalues):
        eligible = np.flatnonzero(q <= q_threshold)
        ranks.append(eligible[np.argsort(-np.abs(effects[eligible]), kind="stable")])
    result = {"de_ranking": "absolute_observed_delta_descending_then_gene_axis",
              "de_q_threshold": float(q_threshold), "de_threshold_operator": "<=",
              "de_empty_condition_policy": "exclude_and_report", "de_evaluated_genes": observed.shape[1]}
    for k in top_ks:
        direction, correlations, counts = [], [], []
        for p, y, rank in zip(predicted, observed, ranks):
            selected = rank[:k]
            counts.append(len(selected))
            if not len(selected):
                continue
            p, y = p[selected], y[selected]
            direction.append(float((np.sign(p) == np.sign(y)).mean()))
            if len(selected) >= 2 and np.std(p) > 0 and np.std(y) > 0:
                correlations.append(float(np.corrcoef(p, y)[0, 1]))
        result.update({"dac_de" + str(k): float(np.mean(direction)) if direction else None,
                       "pcc_de" + str(k): float(np.mean(correlations)) if correlations else None,
                       "de{}_defined_conditions".format(k): len(direction),
                       "de{}_pcc_defined_conditions".format(k): len(correlations),
                       "de{}_gene_counts".format(k): counts})
    return result


def differential_expression_score(predicted_cells, observed_cells, control_cells, *,
                                  family_scope, total_measured_genes,
                                  expression_scale="log1p", q_threshold=0.05,
                                  pseudocount=1e-8):
    """DES for one condition from actual cell-level predictive distributions.

    MWU/BH is recomputed independently for predicted-vs-control and
    observed-vs-control, over every supplied gene. The declared family scope is
    mandatory: passing selected genes cannot recover a full-measured-family
    adjusted p-value. No cells are generated, copied or imputed by this helper.

    Observed DE count n defines the denominator; if more than n predicted DE
    genes exist, retain the n with largest absolute log2 fold change between
    arithmetic mean linear expressions (``expm1`` undoes log1p inputs). Exact
    fold-change ties use gene-axis order. Zero observed DE means undefined DES,
    reported as None, because the manuscript formula divides by n.
    """
    from .data import differential_response

    matrices = [np.asarray(value, dtype=float)
                for value in (predicted_cells, observed_cells, control_cells)]
    if (any(value.ndim != 2 or len(value) < 2 for value in matrices) or
            len({value.shape[1] for value in matrices}) != 1 or not matrices[0].shape[1]):
        raise ValueError("DES needs actual cell matrices with >=2 cells per group on the same gene axis")
    if any(not np.isfinite(value).all() or np.any(value < 0) for value in matrices):
        raise ValueError("DES expression matrices must be finite and nonnegative on the declared scale")
    predicted, observed, controls = matrices
    width = predicted.shape[1]
    if type(total_measured_genes) is not int or total_measured_genes < width:
        raise ValueError("total_measured_genes must be an integer >= the supplied gene count")
    if family_scope not in {"full_measured_genes", "selected_genes"}:
        raise ValueError("Declare family_scope as full_measured_genes or selected_genes")
    if family_scope == "full_measured_genes" and total_measured_genes != width:
        raise ValueError("A full-measured-family DES requires every measured gene in all three cell matrices")
    if expression_scale not in {"log1p", "linear"}:
        raise ValueError("expression_scale must be log1p or linear")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("A finite positive fold-change pseudocount is required")
    _, predicted_q, _ = differential_response(predicted, controls, q_threshold)
    _, observed_q, _ = differential_response(observed, controls, q_threshold)
    observed_de = np.flatnonzero(observed_q <= q_threshold)
    predicted_de = np.flatnonzero(predicted_q <= q_threshold)
    n_observed, n_predicted = len(observed_de), len(predicted_de)
    retained = predicted_de
    if n_predicted > n_observed and n_observed:
        with np.errstate(over="ignore", invalid="ignore"):
            pred_linear = np.expm1(predicted) if expression_scale == "log1p" else predicted
            ctrl_linear = np.expm1(controls) if expression_scale == "log1p" else controls
        if not np.isfinite(pred_linear).all() or not np.isfinite(ctrl_linear).all():
            raise ValueError("Linearized expression overflowed while computing DES fold changes")
        fold_changes = (np.log2(pred_linear.mean(axis=0) + pseudocount)
                        - np.log2(ctrl_linear.mean(axis=0) + pseudocount))
        retained = predicted_de[np.argsort(-np.abs(fold_changes[predicted_de]), kind="stable")[:n_observed]]
    elif n_observed == 0:
        retained = np.array([], dtype=int)
    overlap = len(np.intersect1d(retained, observed_de))
    return {"des": float(overlap / n_observed) if n_observed else None,
            "n_observed_de": n_observed, "n_predicted_de": n_predicted,
            "n_retained_predicted_de": len(retained), "n_matched_de": overlap,
            "observed_de_indices": observed_de.tolist(),
            "retained_predicted_de_indices": retained.tolist(),
            "family_scope": family_scope, "bh_family_size": width,
            "total_measured_genes": total_measured_genes,
            "q_threshold": float(q_threshold), "threshold_operator": "<=",
            "expression_scale": expression_scale, "fold_change_pseudocount": pseudocount,
            "fold_change": "log2_arithmetic_mean_linear_expression_ratio",
            "fold_change_ties": "gene_axis_order", "zero_observed_de": "undefined",
            "cell_counts": {"predicted": len(predicted), "observed": len(observed), "control": len(controls)}}
