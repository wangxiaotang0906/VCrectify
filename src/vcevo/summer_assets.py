"""Import published SUMMER JSON assets without executing upstream code or pickle.

The downloaded assets retain their upstream database licenses. They are not
covered by the license of this independent adapter implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from zipfile import ZipFile

from .types import EvidenceItem


RECORD = "https://zenodo.org/records/14915313"
CHECKSUMS = {"gene_summary.zip": "d14e71bedaece4e09105e4f85b819591",
             "kg.zip": "90977691ebac65e02765e34c31b66bd1"}
LICENSES = {"UniProt": "CC BY 4.0", "Ensembl": "Apache 2.0 (per upstream release)",
            "Gene Ontology": "CC BY 4.0", "CORUM": "CC BY-NC 4.0",
            "STRING": "CC BY 4.0", "Reactome": "CC BY 4.0",
            "BioPlex": "Original data terms; not specified in upstream license table"}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pointer(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _json_member(archive: ZipFile, name: str):
    # No extraction, pickles, or general-purpose deserialization. Exact expected
    # members are read; unexpected files in the archive are not touched.
    member = archive.getinfo(name)
    if member.file_size > 512 * 1024 * 1024:
        raise ValueError("SUMMER archive JSON member is unexpectedly large")
    return json.loads(archive.read(member).decode("utf-8"))


def import_official_summer(summary_zip: str, kg_zip: str, genes: Sequence[str],
                           verify_checksums: bool = True,
                           allow_single_node_fallback: bool = False) -> Tuple[List[EvidenceItem], Dict[str, list], Dict[str, Any]]:
    """Read fixed-version official summaries plus their gene-centered KG provenance.

    ``genes`` must cover both response genes and all candidate intervention
    targets. Missing summaries fail explicitly. The graph keeps full one-hop
    neighbor sets for requested genes, including non-requested neighbor entities.
    """
    genes = tuple(dict.fromkeys(genes))
    if not genes or any(not isinstance(gene, str) or not gene for gene in genes):
        raise ValueError("Provide a nonempty list of gene symbols")
    paths = {"gene_summary.zip": Path(summary_zip), "kg.zip": Path(kg_zip)}
    checksums = {name: _md5(path) for name, path in paths.items()}
    if verify_checksums and checksums != CHECKSUMS:
        raise ValueError("SUMMER archive checksums do not match the pinned Zenodo version 14915313")
    summary_files = {"perturbation": "gene_summary/desc_1hop_pert.json",
                     "response": "gene_summary/desc_1hop_gene.json"}
    single_node_files = {"perturbation": "gene_summary/desc_pert.json",
                         "response": "gene_summary/desc_gene.json"}
    selected_members = {}
    fallbacks = []
    with ZipFile(paths["gene_summary.zip"]) as archive:
        summaries = {role: _json_member(archive, name) for role, name in summary_files.items()}
        for role, contents in summaries.items():
            single_node = _json_member(archive, single_node_files[role]) if allow_single_node_fallback else {}
            for gene in genes:
                selected_members[(gene, role)] = summary_files[role]
                text = contents.get(gene)
                if not isinstance(text, str) or not text.strip():
                    fallback = single_node.get(gene)
                    if isinstance(fallback, str) and fallback.strip():
                        contents[gene] = fallback
                        selected_members[(gene, role)] = single_node_files[role]
                        fallbacks.append({"gene": gene, "role": role, "member": single_node_files[role]})
    missing = {role: [gene for gene in genes if not isinstance(contents.get(gene), str) or not contents[gene].strip()]
               for role, contents in summaries.items()}
    if any(missing.values()):
        raise ValueError("Official SUMMER summaries are missing requested genes: %s" % missing)

    graph = {gene: set() for gene in genes}
    provenance: Dict[str, list] = {gene: [] for gene in genes}
    with ZipFile(paths["kg.zip"]) as archive:
        root = "perturbqa/datasets/kg/"
        for source in ("go", "reactome", "corum", "string", "bioplex"):
            member = root + source + ".json"
            data = _json_member(archive, member)
            mapping = data[0] if source in ("go", "reactome", "corum") else data
            if not isinstance(mapping, dict):
                raise ValueError("Unexpected official KG schema: %s" % member)
            for gene in genes:
                records = mapping.get(gene, [])
                if records:
                    pointer = ("/0/" if isinstance(data, list) else "/") + _pointer(gene)
                    provenance[gene].append({"member": member, "json_pointer": pointer,
                                             "records": records})
                for record in records:
                    if not isinstance(record, list) or not record or not isinstance(record[0], str):
                        raise ValueError("Unexpected KG annotation record in %s" % member)
                    neighbor = record[0]
                    if source in ("reactome", "corum"):
                        neighbor = source + ":" + neighbor
                    # GO IDs already have a namespace; physical neighbors retain
                    # the same symbol namespace as intervention/readout genes.
                    graph[gene].add(neighbor)
        ensembl = _json_member(archive, root + "ensembl.json")
        for identifier, entry in ensembl.items():
            gene = entry.get("name")
            if gene in provenance:
                provenance[gene].append({"member": root + "ensembl.json",
                                         "json_pointer": "/" + _pointer(identifier), "records": entry})
        del ensembl
        uniprot = _json_member(archive, root + "uniprot.json")
        for index, entry in enumerate(uniprot):
            names = entry.get("gene", [])
            if isinstance(names, str):
                names = [names]
            for gene in names or []:
                if isinstance(gene, str) and gene in provenance:
                    provenance[gene].append({"member": root + "uniprot.json",
                                             "json_pointer": "/" + str(index), "records": entry})

    items = []
    for gene in genes:
        for role in ("perturbation", "response"):
            text = summaries[role][gene]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty/non-text official summary for %s" % gene)
            items.append(EvidenceItem(
                id="summer:14915313:%s:%s" % (gene, role), text=text, genes=(gene,),
                source=RECORD + "/files/gene_summary.zip?download=1", kind="summary",
                metadata={"role": role, "summary_model": "Llama3-70B (reported upstream)",
                          "archive_member": selected_members[(gene, role)], "json_pointer": "/" + _pointer(gene),
                          "summary_level": "one_hop" if selected_members[(gene, role)] == summary_files[role] else "single_node",
                          "archive_md5": checksums["gene_summary.zip"],
                          "supporting_kg_archive": RECORD + "/files/kg.zip?download=1",
                          "supporting_kg_md5": checksums["kg.zip"],
                          "supporting_subgraph": provenance[gene],
                          "neighbors": sorted(graph[gene]), "upstream_licenses": LICENSES,
                          "provenance_scope": "Upstream gene-centered source records; not sentence-level citations"}))
    report = {"source": RECORD, "doi": "10.5281/zenodo.14915313", "genes": list(genes),
              "summary_count": len(items), "archive_md5": checksums,
              "integrity_verified": verify_checksums, "upstream_licenses": LICENSES,
              "single_node_fallbacks": fallbacks,
              "graph": "Undirected one-hop neighborhoods from GO, Reactome, CORUM, STRING, BioPlex",
              "observed_experimental_outcomes_imported": False,
              "notes": ["Official prior summaries are LLM-generated; source provenance is not a proof of correctness.",
                        "Knowledge graph and summary assets retain their upstream licenses, including CORUM's non-commercial terms."]}
    return items, {gene: sorted(neighbors) for gene, neighbors in graph.items()}, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import official SUMMER prior assets (no LLM calls)")
    parser.add_argument("--summaries", required=True, help="Downloaded official gene_summary.zip")
    parser.add_argument("--kg", required=True, help="Downloaded official kg.zip")
    parser.add_argument("--genes", required=True, help="JSON array of gene symbols, including perturbation targets")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--allow-single-node-fallback", action="store_true",
                        help="Use an official single-node summary if the official one-hop summary is empty; record every fallback")
    args = parser.parse_args(argv)
    genes = json.loads(Path(args.genes).read_text(encoding="utf-8"))
    items, graph, report = import_official_summer(args.summaries, args.kg, genes,
                                                allow_single_node_fallback=args.allow_single_node_fallback)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("items.json", [asdict(item) for item in items]),
                          ("graph.json", graph), ("provenance.json", report)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Imported %d official summaries for %d genes into %s; no LLM calls made." %
          (len(items), len(genes), output))


if __name__ == "__main__":
    main()
