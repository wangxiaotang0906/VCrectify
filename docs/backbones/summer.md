# SUMMER reasoning adapter

This is an independent implementation of the **summarize → retrieve → answer**
procedure in [Wu et al., ICLR 2025](https://arxiv.org/abs/2502.21290), adapted to
the VCevo manuscript. It uses actual LLM inference when a model is configured.
There is no voting/classifier fallback presented as SUMMER.

The example configuration leaves the HTTP endpoint empty and names Qwen3-8B.
No service is contacted until the endpoint is supplied. A missing endpoint, a
missing gene summary, or an invalid model response produces an explicit error.
An offline reference reasoner used by other examples is a separate backbone;
its results are not SUMMER results or a reproduction of the paper's tables.

## What is preserved and what is adapted

The adapter uses two summaries for each gene: the downstream consequences when
the gene is perturbed (`perturbation`) and the upstream processes that affect
its expression (`response`). Gene proximity is the count of shared neighbors in
an undirected biological knowledge graph. Retrieved examples come only from
already observed experiments in the same cellular context and intervention
modality. Each example includes gene-summary IDs and its observed label.

The default budgets are ten related genes and five examples in each of three
disjoint categories: related perturbation and response, related perturbation
only, and related response only. Exact gene matches also count as related.
Examples for the query condition itself are excluded. Candidate cases are
ordered by the mean reliability of their supporting summaries, with a seeded
tie break. Retrieved summaries are presented in decreasing reliability order.
This implements VCevo's reliability adaptation; it differs from unrestricted
random sampling in the original SUMMER benchmark.

The frozen LLM first answers differential expression (0 or 1). Only if it
predicts differential expression does a second call predict direction (-1 or
+1). This is the three-class mapping in the VCevo appendix. The original
PerturbQA benchmark evaluates DE and direction as separate binary tasks.
The original work reports Llama3-70B summaries, Llama3-8B answering, and three
retrieval trials. This adapter uses the configured fixed model (the manuscript
specifies Qwen3-8B), one retrieval trial, and structured JSON answers. These
choices are explicit and should be reported with results.

## Import the authors' published prior assets

The fixed-version distribution is [Zenodo 14915313](https://zenodo.org/records/14915313).
Download `gene_summary.zip` and `kg.zip` there. The importer verifies the
published archive checksums and reads only named JSON members without extracting
archives or loading pickle:

```text
gene_summary.zip  MD5 d14e71bedaece4e09105e4f85b819591
kg.zip            MD5 90977691ebac65e02765e34c31b66bd1
```

Create a JSON array covering every response gene and every intervention target,
for example `["STAT1", "TP53", "HBG1"]`, then run:

```bash
python -m vcevo.summer_assets \
  --summaries third_party/summer/assets/gene_summary.zip \
  --kg third_party/summer/assets/kg.zip \
  --genes data/required_genes.json \
  --output data/knowledge/summer
```

The output is `items.json` (an array of `EvidenceItem` records), `graph.json`,
and `provenance.json`. All imported summaries are real upstream artifacts; the
import command does not call any language model or load experimental outcomes.
Upstream members `desc_1hop_pert.json` and `desc_1hop_gene.json` contain
`{gene_symbol: summary}` mappings. The importer records their exact archive
member, JSON pointer, and archive checksum. It also stores gene-centered source
records from GO, Reactome, CORUM, STRING, BioPlex, Ensembl, and UniProt, with
pointers into the KG archive. The graph keeps requested genes' full one-hop
neighbor sets, including pathway entities and genes outside the response panel.

The upstream summaries do not provide sentence-level citations. Their recorded
source subgraphs establish provenance, **not** independent verification of every
generated biological claim. New source-attributed facts can instead be processed
by `generate_gene_summaries(items, genes, client)`, which requires explicit
source-ID citations from the configured fixed LLM; persist those outputs before
the experiment and keep calibration outcomes out of summary generation.

The real K562 smoke panel requires 88 distinct symbols (64 readouts and 24
intervention targets). The official one-hop summaries for `HIST1H2AC` and
`HIST1H4C` are empty in both roles. For this panel, add
`--allow-single-node-fallback` to use those genes' nonempty **official
single-node** summaries. The four affected items carry `summary_level=single_node`
and the manifest records every fallback; the remaining 172 summaries are one-hop.
No new biological text is fabricated, and all genes retain their KG neighborhoods.
Without that explicit option the importer fails on incomplete coverage.

For the prepared K562 example, this wrapper derives the required symbol list and
writes exactly the paths used by `configs/k562_txpert_summer.yaml`:

```bash
python scripts/prepare_summer_priors.py --allow-single-node-fallback
```

## Input schema and model configuration

One summary item has this shape; its text must be supplied from an actual source:

```json
{
  "id": "summer:14915313:STAT1:perturbation",
  "text": "<source-backed summary text>",
  "genes": ["STAT1"],
  "source": "https://zenodo.org/records/14915313/files/gene_summary.zip?download=1",
  "kind": "summary",
  "context": "",
  "metadata": {"role": "perturbation", "source_ids": ["<source ID>"]}
}
```

`context=""` means a general biological prior, not a K562 experimental result.
Every query/readout gene also needs the corresponding `response` summary. The
graph JSON maps each gene to adjacent gene/pathway IDs, e.g. two genes connected
to a shared annotated pathway. Symbol alignment is required; the importer does
not silently substitute aliases or missing summaries.

Programmatic HTTP configuration:

```python
from vcevo.backbones.summer import SummerReasoner

reasoner = SummerReasoner.from_config({
    "backend": "http",
    "base_url": "http://127.0.0.1:8000/v1",  # your actual configured server
    "model": "Qwen3-8B",                     # server's actual served model ID
    "api_key_env": "VCEVO_API_KEY",
    "graph_path": "data/knowledge/summer/graph.json",
    "cache_dir": "runs/llm_cache",
    "temperature": 0.0,
    "max_new_tokens": 384,
    "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 on vLLM/SGLang
    "response_retries": 1
})
```

Credentials are read from the named environment variable, never from a config
field or URL. The API is `/chat/completions` with JSON object output mode.
An endpoint that does not support that mode may set `json_mode=false`; returned
text is still validated. Pin the server's model revision and inference software
in a research run; the adapter cannot verify remote model weights from a name.

For Qwen3 served by vLLM, `chat_template_kwargs={"enable_thinking": false}` is
the documented hard switch for compact non-thinking responses; see the
[Qwen deployment documentation](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md)
and [vLLM reasoning documentation](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).
This is a server extension, not a universal OpenAI-compatible field. The client
only sends it when configured; remove it or set it to `null` for a provider that
does not support the extension, and configure non-thinking mode on that provider.
The setting is included in cache and checkpoint identity. The factory defaults
to the HTTP backend and requires an explicit service URL; local inference must
be selected with `backend="transformers"`.

For an already downloaded local checkpoint use `backend="transformers"`,
`model="/path/to/Qwen3-8B"`, and `local_files_only=true` (the default). Local
inference requires the optional `transformers` dependency. Parameters are frozen,
evaluation mode is enabled, and Qwen3 thinking output is disabled to request a
compact structured response. Downloads require explicitly setting
`local_files_only=false`. The application does not do this automatically.

## Evidence boundary and failure behavior

`predict(conditions, genes, knowledge)` receives query metadata, priors,
reliabilities, and `knowledge.memory` containing only revealed observations.
It does not receive an oracle or dataset object. The engine must preserve the
memory gene axis and insert a batch only after predictions and correction.
Fresh validation folds use fresh knowledge states; the frozen model client is
safe to share. Checkpoint loading verifies model and retrieval settings and
restores the retrieval seed. Reproducible remote generation additionally depends
on the server; the client cannot force deterministic server behavior.

Answers contain integer `label`, concise `rationale`, `citations` (summary IDs),
and `case_citations` (historical example IDs). Unknown IDs and unsupported labels
raise an error. The code never interprets malformed output or abstention as
"unchanged." Format retries are bounded. Model rationales and case references
are retained, while only prior summary IDs enter VCevo reliability updates.
Model output is evidence-informed prediction, not a validated causal explanation.

Cache keys include the full evidence prompt, including reliability values and
observed examples, and model generation settings. Corrections therefore invalidate
stale answers naturally. Use a new cache directory when changing weights at an
unchanged local path or when a remote service changes weights under the same ID.

The tests in `tests/test_summer.py` validate two-stage gating, historical-only
retrieval, summary provenance, citation checks, cache invalidation, and an HTTP
round trip through a local **integration fixture**. Those deterministic fixture
responses are not biological model results.

## Attribution and licensing

The [official repository](https://github.com/Genentech/PerturbQA) and its scripts
use the Genentech Non-Commercial Software License 1.0. This independent adapter
does not vendor those scripts or their prompts. The upstream KG and summary
assets retain their original database terms, including **CORUM CC BY-NC 4.0**;
they must not be described as uniformly permissive or covered by this project's
code license. Keep the original distribution README with downloaded assets and
cite both SUMMER and the contributing databases in released experiments.
