# Replogle K562 data contract

The runnable example uses actual K562 essential CRISPRi cells from **Replogle et al. (2022)**. It is a small, seeded real-data experiment for checking the complete framework; it is not a reproduction of the manuscript's reported scores.

## Source and acquisition

Use `K562_essential_raw_singlecell_01.h5ad` from the authors' [processed Perturb-seq deposition](https://doi.org/10.25452/figshare.plus.20029387), associated with [Mapping information-rich genotype–phenotype landscapes with genome-scale Perturb-seq](https://doi.org/10.1016/j.cell.2022.05.013). Select the **raw single-cell K562 essential** file. The deposition also contains genome-wide K562, RPE1, pseudobulk and gemgroup Z-normalized data; these are different inputs. The authors list this deposition under CC BY 4.0. Cite their paper and data deposit when reusing it.

Place the file at:

```text
dataset/Replogle/K562_essential_raw_singlecell_01.h5ad
```

The local file inspected for this example has 310,385 cells × 8,563 source features, takes 10,661,879,995 bytes, and stores dense `float32` raw UMI counts in `X`. `obs/gene` supplies single-target perturbations; its `non-targeting` category contains 10,691 control cells. There are 2,057 other target categories before minimum-cell filtering. `var/gene_name` supplies gene symbols, and `var/gene_id` supplies Ensembl identifiers. The loader handles this file's legacy categorical encoding without requiring AnnData, and also supports modern categorical encodings and CSR `X`.

The download and these cells are not bundled into the source distribution. The source-file size and absolute input path are recorded in each generated manifest; users should verify a downloaded file against the provider's checksum independently.

## Executable preprocessing

After installing the package, run:

```bash
python -m vcevo.data dataset/Replogle/K562_essential_raw_singlecell_01.h5ad data/processed/k562_smoke
```

Default settings retain at most 24 perturbation conditions, 32 cells per condition, 128 controls and 64 response genes. Eligible conditions have at least eight source cells. Conditions and cells are sampled uniformly without replacement, using separate seeds. Conditions with fewer than 32 cells retain all their available cells.

1. Decode the source annotations and identify K562 `non-targeting` controls.
2. Collapse duplicate gene symbols by summing their **raw counts**, retaining all original feature indices and Ensembl IDs in the manifest. In the inspected source, `TBCE` and `HSPA14` each have two source columns, leaving 8,561 unique genes.
3. Normalize each selected cell to a library size of 10,000 over **all source genes**, then apply `log1p`. The source is already filtered to expressed features by its authors; no claim is made that its library includes unprovided genes. Gene filtering happens after normalization. Non-integer input and zero-library cells fail explicitly.
4. Rank genes by the variance of the normalized sampled **controls only**, with stable source-index tie breaking. No perturbation or held-out expression is used to select response genes. Perturbation targets are represented separately from response genes and are not forcibly added to the response axis.
5. Randomly split whole perturbation conditions into initial training, candidate pool, test and validation. Defaults are 8 / 8 / 4 / 4, respectively. A separate split RNG is derived from the condition seed. A condition and its cells cannot occur in more than one split.
6. For every retained condition, compare its sampled cells with the sampled controls using the two-sided Mann–Whitney U test, which is the independent-sample Wilcoxon rank-sum test. Use the asymptotic distribution with tie and continuity corrections. Apply Benjamini–Hochberg adjustment independently per condition across **all 8,561 unique source genes**, and only then project to the response axis.
7. Store the mean difference in normalized log expression, adjusted q-values, cells, and ternary labels `sign(delta) × I[q < 0.05]`.

`bh_family="selected_genes"` is an explicit alternative Python option and changes the family of hypotheses. It must not be silently mixed with default runs. The gene selection here is a simple control-only variance ranking, not Scanpy's `seurat_v3` HVG procedure.

## Statistical scope and information boundaries

The current example pools controls within the K562 cellular context. Its rank-sum test treats cells as independent; it does not estimate a donor-level effect, adjust gemgroups, model batch covariates or claim biological-replicate inference. If a study requires within-gemgroup matching, replicated pseudobulk inference or a different normalization protocol, provide an alternative preprocessing adapter and record that protocol. These choices are scientifically meaningful and should appear in a full experimental report.

The prepared artifact contains all outcomes because it serves as a local retrospective experimental oracle. The framework must expose pool outcomes only **after** selecting conditions, and reserve test/validation outcomes for their stated purposes. A saved split prevents overlap; it does not by itself enforce access control. Backbones receive conditions and currently available observations through their contracts, never the complete prepared dataset.

The default small sample has limited power after full-family multiple-testing correction. Its high share of zero labels is expected and should not be interpreted as evidence of biological absence. The verified local default artifact has 760 perturbed cells, 128 controls and 1,536 condition–response labels, comprising 25 negative, 1,489 non-significant and 22 positive labels. This label histogram is a preprocessing audit, not an evaluation result.

## Artifact schema

The prepared directory contains:

| File | Contents |
| --- | --- |
| `manifest.json` | Schema version, ordered gene symbols, condition identity/context/targets, disjoint split lists, source metadata, normalization and DE definitions, seeds, original cell rows, gene/Ensembl mappings and retained counts |
| `dataset.npz` | Control mean and selected control cells; per-condition normalized selected cells, continuous delta, q-values and `-1/0/+1` labels |

The NPZ contains only numerical arrays and is opened with `allow_pickle=False`. `load_prepared()` checks the contract, including axis consistency, finite values, valid labels/q-values, and a disjoint exhaustive condition partition. `Observation` copies arrays and makes them read-only.

```python
from vcevo.data import load_prepared

dataset = load_prepared("data/processed/k562_smoke")
print(dataset.genes)
print(dataset.splits)
```

## Scaling beyond the smoke run

`prepare_k562()` accepts `None` for each sampling cap and supports integer or fractional split sizes. For example, use all eligible conditions and up to 128 cells per condition while preserving a fixed 2,000-gene response axis:

```python
from vcevo.data import prepare_k562

dataset = prepare_k562(
    "dataset/Replogle/K562_essential_raw_singlecell_01.h5ad",
    "data/processed/k562_research",
    n_genes=2000,
    max_conditions=None,
    max_control_cells=2048,
    max_cells_per_condition=128,
    initial_size=0.10,
    validation_size=0.10,
    test_size=0.15,
    seed=17,
    cell_seed=18,
)
```

This example is a configurable research protocol, not an assertion that those settings reproduce a table in the paper. Align the final protocol, model initialization, resources, evidence priors and evaluation with the experiment being claimed.

Source I/O reads at most `batch_rows` rows at a time; it never reads or densifies the whole source matrix in one call. Full-gene normalized controls and the current perturbation group are resident during testing. The resulting selected-gene cells for retained conditions are held in `PreparedDataset`; therefore choosing all cells and all genes can still require many gigabytes of memory and output storage. All 10,691 controls across 8,561 genes alone use about 366 MB in `float32`, before temporary rank-sum arrays. The large configuration is not executed automatically.
