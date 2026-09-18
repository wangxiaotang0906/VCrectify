# Evaluation protocol

Evaluation compares model outputs with held-out conditions. Outcomes used here never enter fitting, acquisition, reliability calibration or replay. The engine saves and restores predictive adapter state around evaluation so that reporting does not advance a stochastic adapter's acquisition/training RNG.

This document distinguishes the appendix's equations from the implementation conventions required where the manuscript leaves a case unspecified. A smoke run on selected genes and a few conditions is not a reproduction of a full benchmark table.

## Mean-response metrics

The numerical contract predicts a condition-by-gene matrix of mean perturbation effects, `predicted_delta`. Observations use mean perturbed expression minus matched control mean on the same normalized expression scale. Because the same control mean cancels, expression-profile RMSE/MAE equal their effect-vector counterparts.

| Metric | Per-condition definition | Across conditions |
| --- | --- | --- |
| RMSE | Square root of mean squared gene error | Arithmetic mean of condition RMSEs; not pooled RMSE |
| MAE | Mean absolute gene error | Arithmetic mean |
| DAC | Fraction of genes with equal predicted and observed effect signs | Arithmetic mean |
| PCC | Pearson correlation across genes between predicted and observed effects | Mean over conditions with defined PCC; defined-condition count also reported |

DAC uses `sign(delta)`, including exact zeros. It does not use significance-filtered ternary labels. A small nonzero observed effect still has a positive or negative sign even if its q-value is not significant.

PCC is undefined when either compared vector is constant. Such conditions are excluded from the correlation mean and counted explicitly. If none remain, PCC is `null`, not an invented zero or perfect correlation. Input NaNs and infinities are rejected.

## DE20 and DE40

`deg_subset_metrics(predicted_delta, observed_delta, observed_qvalues)` evaluates DAC and PCC on observed response subsets. Prediction errors or predicted significance never choose the evaluated genes.

The appendix describes the strongest observed DE responses but does not specify ranking ties or what happens when fewer than 20/40 significant genes exist. The implemented conventions are therefore explicit:

1. Eligible genes have supplied **observed** q-values `<= 0.05`.
2. Rank eligible genes by descending absolute observed mean log-expression change. Exact magnitude ties follow the common gene-axis order.
3. Reuse that single ranking for both subset sizes, making DE20 a subset of DE40.
4. If fewer than the requested number are eligible, use all eligible genes. Report the actual number for every condition.
5. A zero-DE condition has undefined subset metrics and is excluded. PCC additionally requires at least two genes and nonconstant compared vectors. Both denominator counts are reported.

Each subset metric averages its valid **condition-level** scores, not a pooled collection of selected genes. This matters when conditions have different numbers of significant genes.

The supplied q-values retain their preprocessing family. For the default K562 artifact, they were adjusted over all 8,561 unique source genes before selecting 64 readouts. DE20/DE40 in that smoke run nevertheless select only among its 64 supplied readouts: they are not the strongest 20/40 genes of the full measured transcriptome. Preserve the dataset's `de_test.family_size` and gene-axis metadata when reporting these numbers. These conventions should be aligned with the final manuscript protocol before claiming numerical reproduction.

## Perturbation discrimination score (PDS)

PDS needs mean effects, so it is available through the standard numerical contract. For each prediction, rank all observed effects from evaluated conditions in the **same cellular context** by ascending L1 distance. If the correct counterpart has rank `r` among `N` conditions, its score is:

```text
PDS = 1 - (r - 1) / N
```

This follows the appendix denominator `N`, not `N - 1`. Conditions receive equal weight in the final arithmetic mean. Exact distance ties receive their average rank, an explicit convention because the appendix does not define ties. The result is invariant to reordering tied candidates.

PDS depends on both the response-gene axis and the comparison pool. For four conditions its minimum is 0.25 and an uninformative average-rank score is 0.625. A context with one condition scores 1 by the stated formula but provides no discrimination evidence. Report the number of evaluated conditions per context; do not compare a four-condition smoke PDS directly with a large benchmark pool.

## Differential expression score (DES)

**DES requires actual model-predicted cell-level expression distributions.** A backbone providing only mean effects has no DES under this implementation. The framework must report it as unavailable, not copy a mean vector into artificial cells, borrow observed dispersion or substitute the reasoning model's DE labels.

For one condition, call:

```python
from vcrectify.metrics import differential_expression_score

result = differential_expression_score(
    predicted_cells,
    observed_cells,
    matched_control_cells,
    family_scope="selected_genes",
    total_measured_genes=8561,
    expression_scale="log1p",
)
```

All three arrays must have at least two real/predicted cells on the same gene axis and the declared nonnegative expression scale. The helper does not generate, resample or clip cells. A model's generation procedure, sample count, sampling seed and any output transformation belong in its adapter and experiment provenance.

The runner calls the optional `predict_cells` capability when matched control cells are available. An unconstrained decoder can produce negative expression after a short smoke training run. For this case it reports `des: null` with `des_status: negative_generated_expression_on_log1p_scale`, without silently clipping predictions. This limitation does not invalidate mean-effect error metrics; it does prevent interpreting that output as a valid nonnegative expression distribution for the implemented DES protocol.

The helper applies the appendix definition:

1. Independently compare predicted cells against controls, and observed cells against controls, using two-sided Mann–Whitney U / independent-sample Wilcoxon rank-sum tests, with asymptotic tie and continuity correction.
2. Independently perform Benjamini–Hochberg adjustment across every **supplied** gene for each comparison. Mark DE at q `<= 0.05` as specified for DES in the appendix.
3. Let `n` be the observed DE count. If predicted DE count is at most `n`, retain the complete predicted set. Otherwise retain the `n` predicted DE genes with greatest absolute log fold change.
4. Score the intersection of the retained predicted and observed DE sets divided by `n`.

Absolute fold-change ranking uses `log2((mean_linear_predicted + 1e-8) / (mean_linear_control + 1e-8))`. For `expression_scale="log1p"`, `expm1` first recovers the corresponding linear normalized expression. For `"linear"`, inputs are already on that scale. The pseudocount is configurable and recorded. Exact fold-change ties use gene-axis order. These transform and tie conventions are specified here because the manuscript names log fold change without defining its estimator.

When the observed DE set is empty, the manuscript formula has a zero denominator. The helper returns `des: null` and the observed/predicted counts; it does not assign zero or one. To aggregate multiple conditions, average defined condition scores with equal condition weights and report the number of defined and undefined conditions.

### BH family and selected readouts

`family_scope` and `total_measured_genes` are mandatory. `"full_measured_genes"` is accepted only if every measured gene is present in all three matrices. Use `"selected_genes"` when, for example, the artifact contains 64 of 8,561 measured genes. The result then declares that its BH family has 64 genes and its DES is a **selected-gene DES**.

Existing full-family observed q-values cannot be combined with newly computed selected-family predicted q-values without changing the protocol. DES therefore recomputes both sides over the same supplied family. This can yield a different observed DE set from DE20/DE40, which use the artifact's original full-family q-values. Both families must remain visible in the report. A selected-gene DES cannot be labelled a full-transcriptome or official challenge result.

Cell-level tests in the current K562 example pool the sampled K562 controls and treat cells as independent. They do not supply biological-replicate inference or gemgroup adjustment; see [the data contract](data.md).

## Reasoning classification

Reasoning labels are validated as `-1 / 0 / +1` **before** integer conversion. The class `0` means the observed significance criterion was not met; it does not prove the true biological effect is absent.

Classification metrics pool the evaluated condition–gene queries and calculate class-specific precision, recall and F1 for the fixed three classes: down, non-DE and up. Macro-F1 is the arithmetic mean of all three class F1 values. A class with no true or predicted examples gets F1=0 and remains in the macro average. Therefore an all-non-DE evaluation with perfect non-DE predictions has accuracy 1 but macro-F1 1/3. Per-class supports are reported so that this convention and imbalance remain visible.

## Reporting checklist

Report the gene axis, conditions per context, dataset split seed, control/cell sample counts, preprocessing normalization, statistical hypothesis family, ranking conventions and undefined-metric denominators alongside model settings. Distinguish actual measured K562 subsamples from synthetic contract tests. For uncertainty estimates, repeat the declared experiment across independent seeds; one successful run does not provide a confidence interval.
