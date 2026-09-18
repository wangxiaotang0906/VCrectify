# TxPert numerical backbone

The adapter imports **the original `gspp.models.txpert.TxPert` class** from
[valence-labs/TxPert](https://github.com/valence-labs/TxPert). It does not replace
TxPert with a small MLP. It exposes the standard VCevo numerical contract so
another numerical model can be selected without changing the acquisition loop.

## Source and licensing

Validated upstream revision: `08d82eea86746b044cf7531f4ec8c5f60e1cb73f`.
The downloaded GitHub ZIP has SHA-256
`1bba7c8c1672a1cec7fa030ec6189b9c0dd1b0e3f4a07c23c2387bc31829f03d`.
The adapter records SHA-256 hashes of the actual imported model file, the upstream
Python source tree, and each prior graph in its metadata. Third-party source, graphs and checkpoints remain external
assets; they are not redistributed as VCevo source.

**TxPert has a separate Recursion Non-Commercial End User License Agreement.**
See the upstream [license](https://github.com/valence-labs/TxPert/blob/main/license.pdf)
and [README](https://github.com/valence-labs/TxPert#license) for its full terms.
The VCevo repository does not relicense TxPert. We used the TxPert AI model,
available from Recursion Pharmaceuticals, with software documentation at
[the official repository](https://github.com/valence-labs/TxPert).

The model is described in [Wenkel et al., Nature Biotechnology
(2026)](https://doi.org/10.1038/s41587-026-03113-4). Public source includes
GAT, Exphormer and two-public-graph Exphormer configurations. The published best
model also uses proprietary graphs, so the public example is **not a reproduction
of the best published four-graph result**.

## Install the independent upstream model

Use a separate environment for reproducible research. Upstream specifies Python
3.12 / PyTorch 2.6.0; its own installation instructions install its full research
environment. This adapter needs only the model import chain, not the upstream
Lightning data loader or automatic asset downloader.

```bash
git clone https://github.com/valence-labs/TxPert.git third_party/TxPert-main
git -C third_party/TxPert-main checkout 08d82eea86746b044cf7531f4ec8c5f60e1cb73f
```

Install PyTorch/torchvision for your system first, then:

```bash
python -m pip install torch-geometric==2.6.1 omegaconf==2.3.0 timm==1.0.15 pandas
# Choose the PyG wheel index matching YOUR PyTorch and CUDA combination.
# This example is for PyTorch 2.6 / CUDA 12.4:
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

`timm`/`torchvision` and `torch-scatter` are needed because the upstream module
imports all variants even when using GAT. No modifications or fake modules are
injected into the upstream source. Missing dependencies or assets raise errors.

The local acceptance run also validated the imported model on Windows, Python
3.9.13, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124 and torch-scatter
2.1.2+pt25cu124. This is an observed compatibility result, not upstream's declared
environment. Local optional dependencies were installed into
`.runtime/txpert_deps`; set `dependency_paths: [".runtime/txpert_deps"]` in the
numerical configuration, or add that directory to `PYTHONPATH`.

## Public graph assets

The source archive already contains:

| Prior | Path relative to the upstream source | Size |
|---|---|---:|
| GO similarity | `data/graphs/go/go_top_50.csv` | 12,814,602 bytes |
| STRING v11.5 | `data/graphs/string/v11.5.parquet` | 38,828,749 bytes |

The default example uses the actual upstream GO prior. CSV accepts
`source,target,importance` or `regulator,target,weight`; parquet accepts the same
columns. The adapter retains edges among the declared intervention vocabulary and
the strongest `top_k` incoming edges per target. It reports isolated targets;
GAT adds self loops as configured in upstream. An entirely empty retained graph
fails explicitly. GO similarity does not establish a signed causal effect, and
is never converted into an observed directional label.

Graphs are fixed biological priors. The allowed candidate/held-out **identities**
can appear in their node vocabulary, while expression labels enter the numerical
model only through revealed observations. Do not build the prior graph using
held-out or future expression measurements.

## VCevo adaptation details

- The upstream basal MLP, graph perturbation encoder, latent addition and decoder
  execute unchanged. Default `variant: gat` selects upstream GATv2.
- The example uses LayerNorm and smaller widths than upstream's paper config so
  small continual batches remain valid. Widths, depth, dropout and epochs are
  explicit configuration. This is a **VCevo training configuration**, not an
  official pretrained checkpoint.
- Initialization trains from scratch on D0. For each condition, target cells and
  context-matched control cells are sampled, preserving the complete gene axis.
  The native TxPert loss uses `mse_weight=1`, which is its public GAT default.
- Each condition's cell loss is averaged separately. Selected and replay
  conditions then receive total weights `eta` and `1-eta`, independently of their
  cardinalities. Gradients accumulate over one condition-balanced epoch before
  each Adam step. Empty replay yields selected-only training; an empty selected
  set does not update the model or RNG.
- `cells_per_condition` limits the Monte Carlo cell sample per training epoch;
  all available controls are used during inference. `predict` returns the mean
  of `predicted_expression - matched_control_expression`. `predict_cells` returns
  the actual per-control post-perturbation outputs for distribution metrics.
- The built-in example is single-context K562. Constructor controls must be
  matched to that context and use the same normalized gene axis. Passing only a
  control mean is supported as an explicit approximation and recorded as
  `mean_control_approximation`; the supplied K562 CLI passes control cells.
- `fresh(seed)` rebuilds random weights for LOPO calibration without importing
  previously fitted parameters. Checkpoints include model, optimizer and RNG
  states and reject changed genes, targets, graph hashes, controls, model source,
  architecture or training configuration. Source relocation and device migration
  are allowed.

Official pretrained checkpoints are available from [Zenodo
15420279](https://doi.org/10.5281/zenodo.15420279): checkpoints ZIP approximately
292 MB; single-cell-line cache approximately 678 MB; cross-cell-line cache
approximately 1.75 GB. They are not automatically downloaded or loaded. Their
training conditions may overlap VCevo's future/test conditions; an audited split
and exact feature mapping would be needed before using them in continual-learning
experiments.

## Python example

```python
from vcevo.backbones.txpert import TxPertBackbone
from vcevo.data import load_prepared

data = load_prepared("data/processed/k562_smoke")
model = TxPertBackbone(
    genes=data.genes,
    targets=sorted({t for o in data.observations.values() for t in o.condition.targets}),
    source_dir="third_party/TxPert-main",
    graph_path="third_party/TxPert-main/data/graphs/go/go_top_50.csv",
    control_cells=data.control_cells,
    device="cpu", fit_epochs=2, update_epochs=1,
    hidden_dim=32, latent_dim=16, graph_hidden_dim=16, graph_layers=1,
    batch_size=4, cells_per_condition=4,
)
initial = [data.observations[k] for k in data.splits["initial"]]
model.fit(initial, data.control_mean)
delta = model.predict([initial[0].condition], data.control_mean)
```

For the public multi-graph architecture, use `variant="exphormer"` and pass
`graph_paths={"go": ".../go_top_50.csv", "string": ".../v11.5.parquet"}`
instead of `graph_path`. Reading STRING parquet additionally requires `pyarrow`.
This uses the official multi-graph Exphormer implementation; GPU memory and runtime
increase with graph and hidden dimensions.

## Validation performed

The real prepared K562 smoke subset has 64 response genes, 24 intervention
targets and 128 matched controls. The upstream GO subgraph contains 32 edges and
covers all 24 targets. On CPU, the official GAT completed initialization, selected
plus replay updating, per-cell and mean inference, fresh initialization and exact
checkpoint prediction roundtrip. The official Exphormer also completed a CUDA
fit and finite `(1,64)` prediction using the same GO prior. Integration tests in
`tests/test_txpert_adapter.py` additionally check RNG/optimizer continuation,
empty-update invariance, eta weighting and rejection of incompatible checkpoints.

These are execution checks on a small subset. They establish neither manuscript
metric reproduction nor full-dataset performance.
