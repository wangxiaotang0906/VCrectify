# Implementation and execution record

**Validated locally on 2026-09-18.** These are software and data-integration checks, not a claim to reproduce the manuscript's numerical tables.

## Completed checks

| Check | Observed result |
|---|---|
| Complete local test suite | **81 passed**, 8.41 seconds; four upstream torchvision/Pillow deprecation warnings |
| Static Python checks | `python -m pyflakes src tests scripts examples` passed |
| Synthetic framework demo | Four acquisition rounds completed |
| Real K562 preparation | 64 readouts, 24 perturbation conditions, 128 controls and 760 perturbed cells |
| Official TxPert GAT | Real fit, correction, per-cell inference and mean-effect inference on K562 |
| Official Exphormer | Additional adapter-level CUDA fit and finite prediction check |
| TxPert checkpoint recovery | Resumed and uninterrupted runs have byte-identical acquisition logs and held-out prediction JSON |
| Continual ablations | Adjudicated, Update-all and Random-correction each completed four rounds with independent online acquisition |
| SUMMER prior import | Official archive checksums verified; 176 role summaries cover all 88 required genes |
| SUMMER HTTP integration | Full framework tested with a clearly labelled local HTTP fixture, including cache and resume |
| Live Qwen3-8B | **Not run: service URL intentionally blank** |
| Packaging | Wheel built, installed into an isolated workspace target, and its demo executed successfully |

CI configuration runs the core suite on Python 3.9 and 3.11 on Ubuntu. A clean source-only checkout was also validated locally: **75 passed, 6 skipped**; the six official TxPert integration checks require separately installed upstream source. See the repository's [Actions page](https://github.com/wangxiaotang0906/VCrectify/actions) for the current remote CI status.

## Exact local real-data run

Input was `dataset/Replogle/K562_essential_raw_singlecell_01.h5ad`: 310,385 cells and 8,563 original features. Duplicate symbols `TBCE` and `HSPA14` were merged at raw-count level, leaving 8,561 unique genes. Library-size normalization and log1p precede control-only gene selection. BH correction for observed labels uses all 8,561 genes before projection to the 64-readout panel.

The fixed split is 8 initial / 8 candidate / 4 validation / 4 test conditions. The two training partitions are equal, and all variants use the same starting split and seed. No test or unacquired outcomes enter model fitting, reliability calibration or selection. Source row indices and mappings are preserved in the prepared manifest.

The actual TxPert adapter imports the official class from revision `08d82eea86746b044cf7531f4ec8c5f60e1cb73f`. Its source GO graph retains 32 edges covering the 24 declared targets. Smoke configuration: native GAT, LayerNorm, hidden width 32, latent width 16, graph width 16, one graph layer, three initialization epochs, two correction epochs, and four sampled cells per condition per epoch. Predictions average all 128 supplied controls. This short schedule is intended to exercise code, not train a competitive predictor.

Re-run the full local numerical acceptance protocol in a new directory:

```bash
python scripts/validate_acceptance.py --output runs/acceptance_new
```

The executed output is `runs/acceptance/acceptance.json`; the archive of source code deliberately excludes local data and trained artifacts. Observed correction counts were:

| Policy | Acquired conditions | Numerical corrections | Knowledge corrections |
|---|---:|---:|---:|
| Adjudicated | 8 | 1 | 0 |
| Update-all | 8 | 8 | 8 |
| Random-correction | 8 | 1 | 0 |

The reasoner for these numerical acceptance runs is `reference_observed_frequency`, **not SUMMER**. Counts are determined by the gates; no threshold was altered to force a positive-looking result. The Update-all run exercises both correction paths. The independent tests cover all four adjudication actions and knowledge corrections.

Adjudicated final numerical values on this small test panel: RMSE 1.278541, DAC 0.484375, PCC 0.020502 and PDS 0.625. These poor short-training values are reported as execution evidence, not a claim of method quality. The reference reasoner predicts mostly non-DE: accuracy 0.960938 versus macro-F1 0.326693, with supports down/non-DE/up = 5/246/5. Accuracy alone would be misleading.

DES is `null` in these TxPert smoke results because the unconstrained decoder emits negative log1p-scale expression. The code neither clips the outputs nor repeats a predicted mean to manufacture cells. The true cell-distribution DES implementation is separately tested on valid distributions. Only two of four held-out conditions have significant genes in this selected panel, with seven and three genes, so DE20 and DE40 select identical, smaller subsets. These coverage counts are retained in metrics.

## SUMMER readiness and remaining dependency

The genuine priors cover 64 readouts plus 24 targets. The authors' one-hop summaries are empty for `HIST1H2AC` and `HIST1H4C`; four role summaries explicitly use the same authors' nonempty single-node versions. The other 172 are one-hop. Archive members, KG records, checksums and fallback metadata are preserved. Source provenance does not certify that every LLM-generated summary is biologically correct.

The final offline preflight reports one issue only: missing `VCRECTIFY_LLM_BASE_URL`. The model name is `Qwen3-8B`, and `VCRECTIFY_API_KEY` is the credential environment variable. Preflight estimates at most **14,336 calls before cache hits and format retries** for the smoke configuration, assuming both stages for every gene. Exact LOPO and full candidate rescoring explain this count. It is not a billed-call measurement.

After configuring the endpoint, run:

```bash
python -m vcrectify doctor --config configs/k562_txpert_summer.yaml
python -m vcrectify run --config configs/k562_txpert_summer.yaml
```

The server's exact model identifier, revision, JSON output behavior and Qwen-specific chat-template option still need live validation. No live LLM accuracy, throughput, cost or paper reproduction is claimed. Local Transformers inference is implemented but was not exercised on this host.

## Before a paper-scale experiment

Freeze the final full benchmark split, preprocessing, graph versions, statistical scope and validation-selected hyperparameters; choose adequate training and cell sampling budgets; then collect multiple declared seeds and report their uncertainty. These settings cannot be inferred from result tables alone. The runner intentionally keeps the test set out of selection and does not replace an absent optimized recipe with an invented one.

The independent framework's Apache license does not override TxPert's EULA or SUMMER asset licenses. See [third-party notices](../THIRD_PARTY_NOTICES.md). The source repository is [wangxiaotang0906/VCrectify](https://github.com/wangxiaotang0906/VCrectify). Local datasets, trained artifacts and inference caches are excluded; source archives and wheels can be built using the included packaging configuration.
