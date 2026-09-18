"""Import genuine SUMMER priors for a prepared dataset's readouts and targets."""
import argparse
from dataclasses import asdict
from pathlib import Path

from vcevo.artifacts import write_json
from vcevo.data import load_prepared
from vcevo.summer_assets import import_official_summer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/k562_smoke")
    parser.add_argument("--summaries", default="third_party/summer/assets/gene_summary.zip")
    parser.add_argument("--kg", default="third_party/summer/assets/kg.zip")
    parser.add_argument("--output", default="data/knowledge/summer")
    parser.add_argument("--allow-single-node-fallback", action="store_true")
    args = parser.parse_args()
    dataset = load_prepared(args.data)
    genes = sorted(set(dataset.genes) | {g for o in dataset.observations.values() for g in o.condition.targets})
    items, graph, provenance = import_official_summer(
        args.summaries, args.kg, genes, allow_single_node_fallback=args.allow_single_node_fallback)
    output = Path(args.output)
    for name, value in [("items", [asdict(item) for item in items]), ("graph", graph),
                        ("provenance", provenance), ("required_genes", genes)]:
        write_json(output / (name + ".json"), value)
    print("Imported %d genuine summaries for %d genes. No LLM calls made." % (len(items), len(genes)))


if __name__ == "__main__":
    main()
