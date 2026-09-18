# Backbone contract

VCevo has no hard dependency on a particular numerical architecture or LLM. Built-ins are loaded lazily; `module:factory` entries allow external plugins without modifying the loop.

Numerical factories receive keyword arguments `genes`, `targets`, `control_cells` and their YAML options. Reasoning factories receive their YAML options. A complete minimal example is [examples/custom_backbone.py](../examples/custom_backbone.py).

## Numerical virtual cell

`fit(observations, control_mean)` receives only initial training conditions. `predict(conditions, control_mean)` returns an `(N, G)` matrix of mean expression effects on the declared gene axis. `update(selected, replay, control_mean, eta)` receives only adjudicated new conditions and the old reservoir. Preserve the selected model's native continuous objective; normalize each group over conditions before combining them with `eta` and `1-eta`. An empty reservoir gives the new group full weight.

`fresh(seed)` must construct a new, untrained model. It must not copy a checkpoint trained on a LOPO-held-out condition. Biological graphs and perturbation identities are allowed fixed design information; experimental outcomes are not. `state_dict()` and `load_state_dict()` must include model parameters, optimizer, random generator state and configuration checks. The checkpoint format supports tensors, NumPy arrays and JSON-compatible trees without pickle.

The optional `predict_cells` capability supports distribution-dependent evaluation. A model returning only mean effects cannot supply a valid differential-expression test on predicted cells.

## Reasoning virtual cell

`predict(conditions, genes, knowledge)` returns one `KnowledgePrediction` per condition: labels in `{-1,0,+1}`, supporting summary IDs and a rationale for each response gene. IDs must exist in `knowledge.items`. Cases come exclusively from `knowledge.memory`, never from held-out or unacquired outcomes. The reasoning model is fixed; the engine owns reliability weights and observed memory.

`fresh(seed)` resets any learned or retrieval state for each calibration fold. `state_dict()` and `load_state_dict()` are required even for a stateless adapter (return/accept an empty mapping). Preserve all internal sampling state if present. A model service may be nondeterministic despite a fixed seed; record its version, retain the request cache and do not claim cross-server bitwise reproducibility.

## Inference and evaluation

Inference must not mutate knowledge or train weights. Evaluation snapshots and restores both adapter states so inference RNG consumption cannot change the subsequent acquisition trajectory. A plugin with hidden global state or side effects violates this contract. The oracle boundary prevents accidental label access; it is not a security sandbox for hostile plugin code.

The supplied data example is single-context K562. A multi-context numerical plugin must supply matched control distributions for each context rather than using a shared global control mean. Biological input and output axes, identifier mapping, preprocessing and source revisions belong in the run provenance.

Built-ins: [TxPert](backbones/txpert.md), [SUMMER](backbones/summer.md), and the explicitly diagnostic `reference_ridge` / `reference_memory` models.
