# From manuscript to executable loop

The engine implements the formulas in `sections/method.tex`, with explicit engineering choices where the manuscript leaves implementation details open.

## Statistical labels and thresholds

For each perturbation, observed mean effect is the mean perturbed expression minus the matched control mean. The categorical label is the sign of that effect if its BH-adjusted Wilcoxon/Mann–Whitney p-value is strictly below 0.05, otherwise zero. Initial evidence weight is `(1 + correct citations)/(1 + total citations)`, counting a summary at most once per query. Uncited summaries retain weight one.

Each initial condition is left out in turn. Both states are freshly initialized using only the remaining conditions; initial summary calibration uses predictions without their outcomes in memory. The two thresholds are the empirical 90th percentiles of the held-out numerical and reasoning errors. NumPy's default linear quantile interpolation is used. Thresholds remain fixed during acquisition.

## A round

1. Predict every unobserved candidate with the current two states. Score `sqrt(mean(predicted_effect² × [sign(predicted_effect) != reasoning_label]))`.
2. Select `floor(batch_size/2)` highest-scoring conditions, breaking equal-score ties by stable condition ID. Sample the remainder uniformly without replacement. With a one-condition batch this convention yields random exploration, so examples use batches of two.
3. Flush selected predictions, supporting evidence, scores and memory identifiers to a hash-linked audit log **before** revealing any outcomes in that batch.
4. Compute numerical RMSE and `sqrt(mean(observed_effect² × [predicted_label != observed_label]))`. Correct a state only when its error is strictly larger than its threshold.
5. Update the numerical model using selected new conditions and the previous reservoir. Update cited knowledge reliabilities by `eta × mean_agreement + (1-eta) × old_weight` for selected knowledge corrections. Recorded citations and predictions are not recomputed after observing labels.
6. Admit every newly observed condition to case memory and to the condition-level reservoir, then remove it from the candidate pool. Evaluate on the held-out test set and checkpoint.

Random-correction matches the numbers of numerical and knowledge corrections selected by its own gates in the current batch, sampling each set independently. Update-all corrects both states on every acquired condition. All variants perform their **own online acquisition**; a comparison includes downstream changes in evidence selection, not a controlled comparison on identical evidence streams.

## Reproducibility and limitations

The seed, split manifest, gene mapping, normalization, source rows, initial set, candidate pool, test set and model options are recorded. A resume validates dataset/prior fingerprints, configurations and the entire hash-linked log. It restores models, optimizers, knowledge weights, case order, policy RNGs and reservoir state. Checkpoints use JSON trees and typed arrays, not executable pickle objects.

A package rename changes the recorded source and configuration identity. Start a fresh output directory with the renamed package; use the original code and configuration to resume an earlier experiment.

Transactions commit at round boundaries. A crash after a sealed event but before checkpoint completion leaves an explicit incomplete transaction. Preserve that directory and start a new run; automatic rollback/replay is not implemented. Concurrent writers to one output directory are unsupported.

Test metrics are report-only, with model/RNG state restored after inference. The fixed validation split is reserved for externally specified model selection; this runner does not tune hyperparameters or select checkpoints using test metrics. Exact LOPO training and repeated candidate inference are intentionally faithful and can be computationally expensive. Cache hits reduce repeated fixed-LLM requests, but do not justify replacing current-state predictions with stale outputs.

Reliability measures agreement of a cited item's use with experimental labels. It is neither a calibrated probability that a biological fact is true nor a causal attribution score. The three-way transcriptomic target does not validate a free-text mechanism. External fact acquisition and mechanistic expert scoring require additional protocols.
