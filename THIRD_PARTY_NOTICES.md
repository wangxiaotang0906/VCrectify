# Third-party notices and license scope

The root [Apache License 2.0](LICENSE) applies to independently authored VCrectify framework code and documentation. It does **not** relicense external models, model-specific materials covered by another license, datasets, knowledge graphs, gene summaries, generated model artifacts, or dependencies. The complete configured research system includes components with non-commercial terms; it must not be described as an unrestricted Apache-licensed bundle.

The standard Apache license text includes its unmodified copyright-notice template; that template is not an assertion that a third-party package author owns VCrectify. Existing manuscript text and third-party bibliographic material are not assigned a new license by this code release.

## TxPert

- Original implementation: [valence-labs/TxPert](https://github.com/valence-labs/TxPert).
- Validated source revision: [`08d82eea86746b044cf7531f4ec8c5f60e1cb73f`](https://github.com/valence-labs/TxPert/tree/08d82eea86746b044cf7531f4ec8c5f60e1cb73f).
- Controlling upstream license: [Recursion Non-Commercial End User License Agreement](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/license.pdf), locally retained at `third_party/TxPert-main/license.pdf` when the external source is downloaded.
- Imported model source: [`gspp/models/txpert.py`](https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/gspp/models/txpert.py).

The VCrectify adapter imports the original TxPert class from an external checkout. The inspected upstream source was not modified. The adapter adds VCrectify condition-level fitting, correction, replay, state serialization and evaluation interfaces around that class. Local runs and their trained model artifacts were created using TxPert.

The EULA permits its stated non-commercial research, academic and educational purposes and imposes additional conditions. It includes attribution requirements, restrictions on use and redistribution, and terms governing **Derivative Technology** and associated arising intellectual-property rights. In particular, section 7 requires an essentially equivalent license for such derivative technology. The independent VCrectify Apache grant does not override these terms or determine that every model-specific adaptation/output is outside their scope. Preserve the upstream EULA with any material governed by it; the notices here are not a replacement license.

Required TxPert model attribution, with the upstream template completed:

> We used the TxPert AI model, available from Recursion Pharmaceuticals, with software documentation at www.rxrx.ai, pursuant to Recursion Pharmaceuticals' licensing terms at https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/license.pdf. Under this license, Recursion Pharmaceuticals disclaims all representations and warranties with respect to such AI model.

This attribution describes model use; it does not imply sponsorship, endorsement or official status. The source archive, graphs, checkpoints and local TxPert-derived run artifacts are external/local research assets and are excluded from the source package's license grant.

Verified SHA-256 values for the local acceptance inputs:

| Material | SHA-256 |
| --- | --- |
| Downloaded `txpert-main.zip` | `1bba7c8c1672a1cec7fa030ec6189b9c0dd1b0e3f4a07c23c2387bc31829f03d` |
| Upstream `license.pdf` | `4c122113951fc4c74c65a7f13cb2e85e47dfafa18612fad50918f684655721da` |
| Upstream `gspp/models/txpert.py` | `0caefad613d2608cf221a4f5e27516da91d931bc6139dda05db1719d97731d34` |

The adapter additionally records the files it actually imports and graph hashes in run metadata. These hashes identify inspected bytes; they do not replace the upstream license. See [the TxPert integration guide](docs/backbones/txpert.md) for the source and public-graph configuration.

## SUMMER / PerturbQA

The framework's SUMMER adapter is an independent implementation of the published method and the manuscript's three-class/reliability adaptation. It does not copy, import or distribute the official Genentech implementation. Citation: Wu et al., [Contextualizing biological perturbation experiments through language](https://openreview.net/forum?id=5WEpbilssv), ICLR 2025.

The separate [official PerturbQA repository](https://github.com/Genentech/PerturbQA) is governed by the [Genentech Non-Commercial Software License Version 1.0](https://github.com/Genentech/PerturbQA/blob/main/LICENSE.txt). That license is **not** ordinary Apache 2.0. Users obtaining the official code separately receive its own terms.

Published knowledge graph and summary assets are obtained separately from the authors' [Zenodo data distribution, version 1](https://doi.org/10.5281/zenodo.14915313). Its [data README](https://zenodo.org/records/14915313/files/README.md?download=1) states that `kg.zip` and `gene_summary.zip` retain their source-database licenses. They are not uniformly CC BY 4.0 and are not relicensed by VCrectify.

| Source database in the upstream data README | Terms stated there |
| --- | --- |
| UniProt | CC BY 4.0 |
| Ensembl | Apache 2.0 |
| Gene Ontology, 2024-01-17 release | CC BY 4.0 |
| CORUM | **CC BY-NC 4.0** |
| STRING | CC BY 4.0 |
| Reactome | CC BY 4.0 |
| BioPlex | Cited by the README, but no license is specified in that table; consult the original source for the applicable grant |

This table reports the upstream asset distributor's statements; it does not extend permissions beyond the original database terms. Summaries derived from these entries retain the applicable underlying restrictions. CORUM-derived material prevents treating the entire mixed prior collection as unrestricted commercial-use data. User-provided priors and newly generated summaries must also retain their originating provenance and applicable terms.

The local acceptance cache verified the published archive checksums:

| Asset | Bytes | Official MD5 |
| --- | ---: | --- |
| `gene_summary.zip` | 6,417,757 | `d14e71bedaece4e09105e4f85b819591` |
| `kg.zip` | 46,080,292 | `90977691ebac65e02765e34c31b66bd1` |

The unmodified upstream data README is retained in the local cache as `third_party/summer/assets/README.upstream.md` (SHA-256 `2e77a6b623936943af9d3f08e9a8730b89e527b815fbda8e220d429751313a08`). Source release instructions download these assets from their authors rather than embedding the archives in VCrectify. See [the SUMMER adapter guide](docs/backbones/summer.md).

## Replogle K562 expression data

The K562 example uses `K562_essential_raw_singlecell_01.h5ad` from Joseph Replogle and Jonathan Weissman's [processed Perturb-seq data deposition](https://doi.org/10.25452/figshare.plus.20029387), identified by its authors as **CC BY 4.0**. Cite Replogle et al., [Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq](https://doi.org/10.1016/j.cell.2022.05.013), Cell (2022), and the data deposition.

The original H5AD is not included in the framework source distribution. VCrectify's prepared data are transformed/subsampled research artifacts; their manifests document normalization, duplicate-symbol aggregation, sampling, gene selection and statistical analysis. These transformations do not replace the original data attribution or license. Details are in [the data contract](docs/data.md).

## Language models, runtimes and Python dependencies

Language-model weights, tokenizers, hosted inference services and their outputs remain subject to the terms of the selected model/provider. A compatible HTTP API or a supported Python client is not a license grant for a model. Record the actual model identifier/revision and applicable model-card license for every release of model-derived artifacts.

Python libraries and native extensions are installed separately and retain their own licenses and notices. [requirements-tested.txt](requirements-tested.txt) records the minimal validated local versions, including platform-specific optional components; it does not relicense them or function as a complete dependency notice inventory for a redistributed binary environment. Preserve each dependency's own license when redistributing it.

These notices describe the inspected inputs as of 2026-09-18. No upstream source, weight, dataset or license is replaced by this file.
