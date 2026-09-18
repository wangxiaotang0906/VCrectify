"""Validate real TxPert loop recovery; requires locally prepared data and model.

This uses the explicitly configured diagnostic reasoner, not a live SUMMER
service. It never changes adjudication thresholds to force a favorable result.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from vcevo.artifacts import write_json
from vcevo.cli import load_config, run_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/k562_txpert_smoke.yaml")
    parser.add_argument("--output", default="runs/acceptance")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["reasoning"]["name"] != "reference_memory":
        raise ValueError("This acceptance script requires the explicitly diagnostic memory reasoner")
    output = Path(args.output).resolve()
    resumed = output / "resumed"
    direct = output / "uninterrupted"
    run_config(config, resumed, max_rounds=1)
    resumed_summary = run_config(config, resumed, resume=True)
    direct_summary = run_config(config, direct)
    same_log = (resumed / "events.jsonl").read_bytes() == (direct / "events.jsonl").read_bytes()
    same_predictions = (resumed / "heldout_predictions.json").read_bytes() == (direct / "heldout_predictions.json").read_bytes()
    if not same_log or not same_predictions:
        raise AssertionError("Resume differs from uninterrupted execution")
    variants = {"adjudicated": direct_summary}
    for policy in ["update_all", "random_matched"]:
        variant = deepcopy(config)
        variant["run"]["correction"] = policy
        variants[policy] = run_config(variant, output / policy)
    gates = {}
    for policy, directory in [("adjudicated", direct), ("update_all", output / "update_all"),
                              ("random_matched", output / "random_matched")]:
        rounds = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        rounds = [row for row in rounds if row["event"] == "round_completed"]
        gates[policy] = {"rounds": len(rounds),
                         "numerical_corrections": sum(r["correction_count_m"] for r in rounds),
                         "knowledge_corrections": sum(r["correction_count_k"] for r in rounds)}
    result = {"resume_audit_identical": same_log, "resume_predictions_identical": same_predictions,
              "resumed_completed": resumed_summary["completed"], "variants": variants,
              "correction_counts": gates, "live_llm_used": False,
              "purpose": "software acceptance; reduced real-data panel; not a performance claim"}
    write_json(output / "acceptance.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "variants"}, indent=2))


if __name__ == "__main__":
    main()
