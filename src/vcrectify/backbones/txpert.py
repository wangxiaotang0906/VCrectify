"""Adapter around the unmodified, independently licensed official TxPert model.

No architecture is reimplemented here. Source and prior graphs must be supplied
explicitly; there is deliberately no fallback to the reference backbone.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from vcrectify.types import Condition, Observation


class TxPertBackbone:
    """Official TxPert with cell-level native MSE and condition-balanced replay.

    ``targets`` is the declared intervention vocabulary, including unobserved
    candidates. It contains identities, never future expression measurements.
    Prior edges are frozen. ``control_cells`` should contain context-matched
    normalized controls. Omitting it explicitly selects the mean-control
    approximation; the run metadata records that choice.
    """

    name = "txpert"

    def __init__(
        self, genes: Sequence[str], targets: Sequence[str], graph_path=None,
        source_dir="third_party/TxPert-main", seed: int = 0, device: str = "cpu",
        fit_epochs: int = 50, update_epochs: int = 5, lr: float = 1e-3,
        hidden_dim: int = 128, latent_dim: int = 32, graph_hidden_dim: int = 64,
        graph_layers: int = 2, dropout: float = 0.0, batch_size: int = 16,
        top_k: int = 20, control_cells=None, cells_per_condition: int = 32,
        variant: str = "gat", graph_paths: Optional[Mapping[str, str]] = None,
        dependency_paths: Sequence[str] = (),
    ):
        self.genes = tuple(genes)
        self.targets = tuple(sorted(set(targets)))
        if len(set(self.genes)) != len(self.genes) or len(self.genes) < 2 or not self.targets:
            raise ValueError("TxPert requires unique response genes and intervention targets")
        if variant not in {"gat", "exphormer"}:
            raise ValueError("Supported official TxPert variants: gat, exphormer")
        if min(fit_epochs, update_epochs, batch_size, top_k, cells_per_condition,
               hidden_dim, latent_dim, graph_hidden_dim, graph_layers) < 1:
            raise ValueError("Epochs, dimensions and batch sizes must be positive")
        if not np.isfinite(lr) or lr <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid learning rate or dropout")
        if graph_hidden_dim % 2:
            raise ValueError("graph_hidden_dim must be divisible by the two attention heads")
        if graph_paths and graph_path:
            raise ValueError("Pass graph_path or graph_paths, not both")
        paths = dict(graph_paths or ({"prior": str(graph_path)} if graph_path else {}))
        if not paths:
            raise FileNotFoundError("TxPert requires an explicit prior graph CSV or parquet")
        if variant == "gat" and len(paths) != 1:
            raise ValueError("GAT uses one graph; use variant='exphormer' for multiple priors")
        for value in paths.values():
            if not Path(value).is_file():
                raise FileNotFoundError("TxPert prior graph not found: " + str(value))
        source = Path(source_dir).resolve()
        if not (source / "gspp/models/txpert.py").is_file():
            raise FileNotFoundError(
                "Official TxPert source missing at %s. See docs/backbones/txpert.md" % source
            )
        for path in [str(source)] + [str(Path(p).resolve()) for p in dependency_paths]:
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            import torch
            import pandas as pd
            official = importlib.import_module("gspp.models.txpert")
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Cannot import official TxPert. Install PyTorch, torch-geometric, "
                "torch-scatter, omegaconf and timm; see docs/backbones/txpert.md. "
                "Original error: %s" % exc
            ) from exc
        imported_source = Path(official.__file__).resolve()
        if imported_source != source / "gspp/models/txpert.py":
            raise RuntimeError("A different gspp package is already imported: %s" % imported_source)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose device='cpu'")
        self.torch = torch
        self.device = str(torch.device(device))
        self.pert2id = {gene: i for i, gene in enumerate(self.targets)}
        self.control_cells = None if control_cells is None else np.array(control_cells, dtype=np.float32, copy=True)
        if self.control_cells is not None:
            if (self.control_cells.ndim != 2 or self.control_cells.shape[1] != len(self.genes)
                    or len(self.control_cells) == 0 or not np.isfinite(self.control_cells).all()):
                raise ValueError("control_cells must be a nonempty finite matrix on the gene axis")
        self.config = dict(
            genes=self.genes, targets=self.targets, graph_paths=paths,
            source_dir=str(source), seed=seed, device=self.device,
            fit_epochs=fit_epochs, update_epochs=update_epochs, lr=lr,
            hidden_dim=hidden_dim, latent_dim=latent_dim, graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers, dropout=dropout, batch_size=batch_size,
            top_k=top_k, cells_per_condition=cells_per_condition, variant=variant,
            dependency_paths=tuple(dependency_paths),
        )
        self.rng = np.random.default_rng(seed)
        self.graph_metadata = {}
        graph_dict = {}
        for name, path in paths.items():
            frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
            frame = frame.rename(columns={"regulator": "source", "importance": "weight"})
            if not {"source", "target", "weight"}.issubset(frame.columns):
                raise ValueError("Graph needs source/regulator,target,weight/importance columns: " + str(path))
            frame = frame[["source", "target", "weight"]].copy()
            frame = frame[frame.source.isin(self.targets) & frame.target.isin(self.targets)]
            if not np.isfinite(frame.weight.to_numpy(dtype=float)).all():
                raise ValueError("Prior graph has non-finite edge weights")
            frame = frame.sort_values(["target", "weight", "source"], ascending=[True, False, True])
            frame = frame.drop_duplicates(["source", "target"]).groupby("target", sort=False).head(top_k)
            if frame.empty:
                raise ValueError("No prior edges connect declared intervention targets in " + str(path))
            edges = np.stack([frame.source.map(self.pert2id), frame.target.map(self.pert2id)]).astype(np.int64)
            graph_dict[name] = (
                torch.tensor(edges, dtype=torch.long),
                torch.tensor(frame.weight.to_numpy(), dtype=torch.float32), len(self.targets),
            )
            connected = set(frame.source) | set(frame.target)
            self.graph_metadata[name] = dict(
                path=str(Path(path).resolve()), sha256=_sha256(path), edges=len(frame),
                target_nodes=len(self.targets), connected_targets=len(connected),
                isolated_targets=sorted(set(self.targets) - connected),
            )
        # This small structural object satisfies the official graph container API;
        # graph computation is performed by the original TxPert implementation.
        graph = SimpleNamespace(graph_dict=graph_dict, pert2id=self.pert2id)
        pert_args = dict(
            model_type="gnn" if variant == "gat" else "exphormer",
            layer_type="gat_v2" if variant == "gat" else "exphormer_w_mpnn",
            num_layers=graph_layers, hidden_dim=graph_hidden_dim, skip="skip_cat",
            dropout=dropout, num_heads=2, concat=True, add_self_loops=True,
            use_edge_weight=False, use_struc_feat=False,
        )
        if variant == "exphormer":
            pert_args.update(expander_degree=3, add_reverse_edges=True, pos_enc="none",
                             union_edge_type="multihot", edge_feat_map_type="linear")
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            self._seed_torch(seed)
            self.model = official.TxPert(
                input_dim=len(self.genes), output_dim=len(self.genes), adata_output_dim=len(self.genes),
                graph=graph, hidden_dim=hidden_dim, latent_dim=latent_dim,
                use_batch_norm=False, use_layer_norm=True, dropout=dropout,
                omit_cntr=True, device=self.device, mse_weight=1.0,
                cntr_model_args=dict(model_type="mlp", hidden_dim=hidden_dim,
                                     dropout=dropout, use_batch_norm=False, use_layer_norm=True),
                pert_model_args=pert_args,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history = []
        self.fitted = False
        self.metadata = dict(
            implementation="gspp.models.txpert.TxPert", source_file=str(imported_source),
            source_sha256=_sha256(imported_source), source_tree_sha256=_source_tree_sha256(source), variant=variant,
            control_policy="all_matched_controls" if self.control_cells is not None else "mean_control_approximation",
            normalization="layer_norm", objective="native_cell_mse_condition_balanced",
            pretrained_checkpoint=False, graphs=self.graph_metadata,
        )

    def _controls(self, control_mean):
        mean = np.asarray(control_mean, dtype=np.float32)
        if mean.shape != (len(self.genes),) or not np.isfinite(mean).all():
            raise ValueError("Invalid control mean")
        if self.control_cells is not None:
            if not np.allclose(self.control_cells.mean(axis=0), mean, rtol=1e-4, atol=1e-5):
                raise ValueError("control_mean does not match the supplied matched control cells")
            return self.control_cells
        return mean[None, :]

    def _indices(self, conditions):
        indices = []
        for condition in conditions:
            missing = set(condition.targets) - set(self.pert2id)
            if missing:
                raise ValueError("Undeclared perturbation targets: " + ", ".join(sorted(missing)))
            indices.append([self.pert2id[target] for target in condition.targets])
        return indices

    def fit(self, observations: Sequence[Observation], control_mean) -> None:
        if self.fitted:
            raise RuntimeError("fit initializes a fresh model; use update or fresh(seed)")
        if not observations:
            raise ValueError("TxPert fit needs initial observations")
        self._train(observations, (), control_mean, 1.0, self.config["fit_epochs"])
        self.fitted = True

    def predict(self, conditions: Sequence[Condition], control_mean) -> np.ndarray:
        controls = self._controls(control_mean)
        if not conditions:
            return np.empty((0, len(self.genes)), dtype=float)
        output = []
        # Stream means: acquisition must not materialize candidates x controls x
        # genes. Distribution outputs are exposed separately by predict_cells.
        for perts in self._indices(conditions):
            total = np.zeros(len(self.genes), dtype=np.float64)
            for start, value in self._prediction_chunks(perts, controls):
                total += (value.astype(np.float64) - controls[start:start + len(value)]).sum(axis=0)
            output.append(total / len(controls))
        return np.stack(output)

    def predict_cells(self, conditions: Sequence[Condition], control_cells) -> list:
        """Return real post-perturbation outputs for each supplied control cell.

        This is an optional distribution-metric capability. Outputs are not
        replicated condition means and contain no experimental outcomes.
        """
        controls = np.asarray(control_cells, dtype=np.float32)
        if (controls.ndim != 2 or controls.shape[1] != len(self.genes)
                or len(controls) == 0 or not np.isfinite(controls).all()):
            raise ValueError("predict_cells needs finite normalized controls on the gene axis")
        indices = self._indices(conditions)
        output = []
        for perts in indices:
            output.append(np.concatenate([value for _, value in self._prediction_chunks(perts, controls)]))
        return output

    def _prediction_chunks(self, perts, controls):
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(controls), self.config["batch_size"]):
                cntr = torch.tensor(controls[start:start + self.config["batch_size"]], device=self.device)
                predicted, _, _ = self.model(cntr, [perts] * len(cntr), None)
                value = predicted.cpu().numpy()
                if not np.isfinite(value).all():
                    raise FloatingPointError("TxPert produced non-finite predictions")
                yield start, value

    def update(self, observations, replay, control_mean, eta: float) -> None:
        if not self.fitted:
            raise RuntimeError("Call fit before a continual update")
        if not 0 <= eta <= 1 or not np.isfinite(eta):
            raise ValueError("eta must lie in [0, 1]")
        if not observations:
            return
        self._train(observations, replay, control_mean, eta, self.config["update_epochs"])

    def _train(self, observations, replay, control_mean, eta, epochs):
        controls = self._controls(control_mean)
        torch = self.torch
        # When replay is empty, the available selected loss has unit weight.
        groups = [(observations, eta if replay else 1.0), (replay, 1.0 - eta)]
        for records, _ in groups:
            self._indices([record.condition for record in records])
            if any(record.cells.shape[1] != len(self.genes) for record in records):
                raise ValueError("Observation has a mismatched gene axis")
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            self._seed_torch(int(self.rng.integers(0, 2**31 - 1)))
            for _ in range(epochs):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for records, weight in groups:
                    if not records or weight == 0:
                        continue
                    # Separate normalization prevents a larger replay pool from
                    # silently changing eta. Gradients accumulate over a full
                    # condition-balanced epoch before one optimizer step.
                    for start in range(0, len(records), self.config["batch_size"]):
                        chunk = records[start:start + self.config["batch_size"]]
                        ys, xs, perts, lengths = [], [], [], []
                        for record in chunk:
                            n = min(len(record.cells), self.config["cells_per_condition"])
                            yidx = self.rng.choice(len(record.cells), size=n, replace=False)
                            xidx = self.rng.integers(0, len(controls), size=n)
                            ys.append(record.cells[yidx])
                            xs.append(controls[xidx])
                            perts.extend(self._indices([record.condition]) * n)
                            lengths.append(n)
                        x = torch.tensor(np.concatenate(xs), device=self.device)
                        y = torch.tensor(np.concatenate(ys), device=self.device)
                        prediction, means, logvars = self.model(x, perts, None)
                        offset = 0
                        chunk_loss = torch.zeros((), device=self.device)
                        for n in lengths:
                            # Basal MLP has no KL term. The official native loss
                            # is MSE over sampled cells and the complete gene axis.
                            loss, _ = self.model.loss(prediction[offset:offset + n], 0, 0, y[offset:offset + n])
                            chunk_loss = chunk_loss + loss * weight / len(records)
                            offset += n
                        if not torch.isfinite(chunk_loss):
                            raise FloatingPointError("Non-finite TxPert training loss")
                        chunk_loss.backward()
                        total_loss += float(chunk_loss.detach().cpu())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                self.history.append(total_loss)

    def _seed_torch(self, seed):
        # Do not change another backbone's CUDA RNG when this model runs on CPU.
        self.torch.random.default_generator.manual_seed(seed)
        if self.device.startswith("cuda"):
            with self.torch.cuda.device(self.device):
                self.torch.cuda.manual_seed(seed)

    def fresh(self, seed: int):
        config = dict(self.config, seed=seed)
        return type(self)(**config, control_cells=self.control_cells)

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            format_version=1, genes=self.genes, targets=self.targets,
            configuration=self._checkpoint_configuration(),
            graph_sha256={key: value["sha256"] for key, value in self.graph_metadata.items()},
            model={key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()},
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            rng=copy.deepcopy(self.rng.bit_generator.state), history=list(self.history),
            fitted=self.fitted, metadata=copy.deepcopy(self.metadata),
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != 1 or state.get("configuration") != self._checkpoint_configuration():
            raise ValueError("TxPert checkpoint architecture, training configuration, controls or source mismatch")
        if tuple(state["genes"]) != self.genes or tuple(state["targets"]) != self.targets:
            raise ValueError("TxPert checkpoint gene/target order mismatch")
        if state["graph_sha256"] != {key: value["sha256"] for key, value in self.graph_metadata.items()}:
            raise ValueError("TxPert checkpoint prior graph mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.history = list(state["history"])
        self.fitted = bool(state["fitted"])

    def _checkpoint_configuration(self):
        # Relocation and CPU/GPU migration are allowed; semantic changes are not.
        excluded = {"seed", "device", "source_dir", "graph_paths", "dependency_paths"}
        config = {key: value for key, value in self.config.items() if key not in excluded}
        config["source_sha256"] = self.metadata["source_sha256"]
        config["source_tree_sha256"] = self.metadata["source_tree_sha256"]
        config["graph_order"] = tuple(self.graph_metadata)
        config["control_cells_sha256"] = None if self.control_cells is None else hashlib.sha256(
            str(self.control_cells.shape).encode() + self.control_cells.tobytes()).hexdigest()
        return config


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(source) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(source) / "gspp").rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()
