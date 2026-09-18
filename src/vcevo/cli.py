"""Reproducible local entry points; no implicit model or data downloads."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import yaml

from . import __version__
from .artifacts import write_json
from .backbones.reference import MemoryReasoner, RidgeBackbone, reference_knowledge
from .data import load_prepared, prepare_k562
from .demo import synthetic_dataset
from .engine import RunConfig, VCevo
from .registry import numerical_backbone, reasoning_backbone
from .types import EvidenceItem, KnowledgeState


def _expand(value):
    if isinstance(value, str):
        def replace(match):
            name, _, default = match.group(1).partition(":")
            return os.environ.get(name, default)
        return re.sub(r"\$\{([^}]+)\}", replace, value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        if any(k.lower() in {"api_key", "token", "password", "secret"} for k in value):
            raise ValueError("Store credentials in an environment variable, using api_key_env in YAML")
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path):
    path = Path(path).resolve()
    config = _expand(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    root = (path.parent / config.pop("project_root", ".")).resolve()
    path_fields = {"prepared", "source", "output", "items_path", "graph_path", "source_dir", "cache_dir"}
    def resolve(obj):
        if isinstance(obj, dict):
            return {k: str((root / v).resolve()) if k in path_fields and isinstance(v, str) and v
                    else resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj
    config = resolve(config)
    for entry in reversed(config.pop("python_paths", [])):
        directory = (root / entry).resolve()
        if directory.is_dir():
            sys.path.insert(0, str(directory))
    return config


def load_knowledge(config):
    if config.get("name") == "reference":
        return reference_knowledge()
    path = config.get("items_path")
    if not path:
        raise ValueError("SUMMER needs knowledge.items_path with source-attributed summary items")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows["items"]
    items = {}
    for row in rows:
        row = dict(row)
        row["genes"] = tuple(row["genes"])
        item = EvidenceItem(**row)
        if item.id in items:
            raise ValueError("Duplicate knowledge item ID: " + item.id)
        items[item.id] = item
    if not items:
        raise ValueError("Knowledge store is empty")
    return KnowledgeState(items)


def diagnose(config):
    issues = []
    data = None
    prepared = config.get("data", {}).get("prepared")
    if prepared:
        try:
            data = load_prepared(prepared)
        except (OSError, ValueError, KeyError) as exc:
            issues.append("Prepared data: " + str(exc))
    else:
        issues.append("Set data.prepared and run the preparation command first")
    numerical = config.get("numerical", {})
    reasoning = config.get("reasoning", {})
    if numerical.get("name") == "txpert":
        for key in ["source_dir", "graph_path"]:
            if not numerical.get(key) or not Path(numerical[key]).exists():
                issues.append("Missing TxPert " + key)
        for module in ["torch", "torch_geometric", "torch_scatter", "omegaconf", "timm"]:
            if importlib.util.find_spec(module) is None:
                issues.append("Missing TxPert dependency: " + module)
    knowledge = None
    try:
        knowledge = load_knowledge(config.get("knowledge", {}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append("Knowledge: " + str(exc))
    if reasoning.get("name") == "summer":
        if reasoning.get("backend", "http") == "http":
            if not reasoning.get("base_url", "").strip():
                issues.append("Set VCEVO_LLM_BASE_URL (or reasoning.base_url); no service URL is configured")
            if not reasoning.get("model", "").strip():
                issues.append("Set the exact model ID served by the endpoint")
        elif importlib.util.find_spec("transformers") is None:
            issues.append("Install the llm optional dependencies for the local Transformers backend")
        if not reasoning.get("graph_path") or not Path(reasoning["graph_path"]).is_file():
            issues.append("Missing SUMMER graph_path")
        if data and knowledge and not reasoning.get("allow_missing_summaries", False):
            required = {(g, "response") for g in data.genes}
            required.update((g, "perturbation") for o in data.observations.values() for g in o.condition.targets)
            available = {(g, h.metadata.get("role")) for h in knowledge.items.values()
                         if h.kind == "summary" for g in h.genes}
            missing = sorted(required - available)
            if missing:
                issues.append("Missing %d required SUMMER role summaries; examples: %s" % (len(missing), missing[:5]))
    report = {"ready": not issues, "issues": issues, "package_version": __version__,
              "python": platform.python_version(), "numerical": numerical.get("name"),
              "reasoning": reasoning.get("name"), "model": reasoning.get("model")}
    if data:
        run = RunConfig(**config.get("run", {})).validate()
        sizes = {k: len(v) for k, v in data.splits.items()}
        n0, pool, test = sizes["initial"], sizes["pool"], sizes["test"]
        remaining, pool_queries, rounds = run.budget, 0, 0
        while remaining > 0:
            pool_queries += pool
            take = min(run.batch_size, remaining)
            pool -= take
            remaining -= take
            rounds += 1
        # Two stages per gene at most, before cache hits/retries. Calibration
        # performs n0 folds: n0-1 prior queries plus the held-out query each.
        report.update(genes=len(data.genes), splits=sizes,
                      llm_call_upper_bound_before_cache_and_retries=(
                          2 * len(data.genes) * (n0 + n0*n0 + pool_queries + test*(rounds+1))
                          if reasoning.get("name") == "summer" else 0))
        if run.budget > sizes["pool"]:
            issues.append("Acquisition budget exceeds prepared pool size")
        report["ready"] = not issues
    return report


def run_config(config, output=None, resume=False, max_rounds=None):
    report = diagnose(config)
    if not report["ready"]:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(report["issues"]))
    data = load_prepared(config["data"]["prepared"])
    targets = sorted({g for o in data.observations.values() for g in o.condition.targets})
    numerical = numerical_backbone(config["numerical"], data.genes, targets,
                                   control_cells=data.control_cells)
    reasoning = reasoning_backbone(config["reasoning"])
    knowledge = load_knowledge(config["knowledge"])
    output = Path(output or config["output"]).resolve()
    # Credentials are never read here. The manifest stores only their env-var
    # name, model settings and endpoint URL (which must contain no credentials).
    provenance = {"numerical": config["numerical"], "reasoning": config["reasoning"],
                  "knowledge": config["knowledge"], "python": platform.python_version(),
                  "vcevo_version": __version__,
                  "numerical_implementation": getattr(numerical, "metadata", {})}
    dependencies = {}
    for package in ["numpy", "scipy", "h5py", "PyYAML", "torch", "torch-geometric", "torch-scatter"]:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            continue
    provenance["installed_dependencies"] = dependencies
    package_root = Path(__file__).resolve().parent
    source_hash = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        source_hash.update(source.relative_to(package_root).as_posix().encode())
        source_hash.update(source.read_bytes())
    provenance["framework_source_sha256"] = source_hash.hexdigest()
    start = time.perf_counter()
    runner = VCevo(data, numerical, reasoning, knowledge, RunConfig(**config.get("run", {})),
                   output, provenance=provenance, resume=resume)
    history = runner.run(max_rounds=max_rounds)
    summary = {"output": str(output), "round": runner.round, "acquired": runner.acquired,
               "completed": runner.acquired == runner.config.budget,
               "elapsed_seconds_this_invocation": round(time.perf_counter()-start, 3),
               "numerical": numerical.name, "reasoning": reasoning.name,
               "last_evaluation": history[-1] if history else None}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vcevo", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="CPU synthetic contract run; no biological claims")
    demo.add_argument("--output", default="runs/synthetic_demo")
    demo.add_argument("--seed", type=int, default=17)
    prepare = commands.add_parser("prepare", help="Prepare the genuine local K562 H5AD")
    prepare.add_argument("--config", required=True)
    doctor = commands.add_parser("doctor", help="Offline data/model/service preflight (does not call the LLM)")
    doctor.add_argument("--config", required=True)
    run = commands.add_parser("run", help="Execute a configured continual experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--output")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-rounds", type=int)
    run.add_argument("--correction", choices=["adjudicated", "update_all", "random_matched"])
    run.add_argument("--acquisition", choices=["disagreement", "random", "magnitude"])
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            data = synthetic_dataset(args.seed)
            runner = VCevo(data, RidgeBackbone(data.genes, seed=args.seed), MemoryReasoner(),
                           reference_knowledge(), RunConfig(seed=args.seed), args.output,
                           provenance={"synthetic": True, "purpose": "framework engineering check"})
            history = runner.run()
            result = {"output": str(Path(args.output).resolve()), "synthetic": True,
                      "acquired": runner.acquired, "last_evaluation": history[-1]}
            write_json(Path(args.output) / "summary.json", result)
        elif args.command == "prepare":
            config = load_config(args.config)
            options = dict(config["prepare"])
            source, output = options.pop("source"), options.pop("output")
            data = prepare_k562(source, output_dir=output, **options)
            result = {"output": output, "genes": len(data.genes),
                      "splits": {k: len(v) for k, v in data.splits.items()}}
        elif args.command == "doctor":
            result = diagnose(load_config(args.config))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ready"] else 2
        else:
            if args.max_rounds is not None and args.max_rounds < 0:
                raise ValueError("max-rounds cannot be negative")
            config = load_config(args.config)
            for key in ["correction", "acquisition"]:
                if getattr(args, key) is not None:
                    config.setdefault("run", {})[key] = getattr(args, key)
            result = run_config(config, args.output, args.resume, args.max_rounds)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print("vcevo: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
