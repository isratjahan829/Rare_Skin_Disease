# DERM-Net cross-dataset benchmark

Four notebooks that benchmark the DERM-Net architecture — EfficientNet-B4 + ViT-B/16
with multi-scale channel-attention (MSCA) fusion, ~107M parameters — against its own
ablations and against standard baselines, on four dermatology datasets under one
protocol.

| Notebook | Dataset | Modality | Runtime (T4) |
|---|---|---|---|
| `benchmark_DERMNetDual.ipynb` | Rare Skin Disease (650 img, 5 cls) | clinical | 1.5–2.5 h |
| `benchmark_HAM10000.ipynb` | HAM10000 (10,015 img, 7 cls) | **dermoscopy** | 4–6 h |
| `benchmark_Fitzpatrick17k.ipynb` | Fitzpatrick17k (9 partitions) | clinical | 3–5 h |
| `benchmark_DDI.ipynb` | DDI (656 img, biopsy-confirmed) | clinical | 1–2 h |

## What is compared

| Model | Params | Role |
|---|---|---|
| **DERM-Net** | ~107M | proposed |
| DERM-Net (no MSCA) | ~105M | ablation: same backbones, plain concatenation |
| DERM-Net (Eff only) | ~19M | ablation: EfficientNet-B4 alone |
| DERM-Net (ViT only) | ~86M | ablation: ViT-B/16 alone |
| ResNet-50 | ~24M | baseline |
| DenseNet-121 | ~7M | baseline |

## The row that decides the paper

**`DERM-Net` vs `DERM-Net (no MSCA)`.**

Beating ResNet-50 may only show that two large pretrained backbones beat one small one —
a result nobody disputes. Only the comparison against plain concatenation of the *same*
two backbones tests the fusion block, which is what the architecture actually contributes.
Each notebook prints this explicitly:

```
MSCA fusion block vs plain concatenation: +X.XX pp, Holm p = ...  ->  supported / not supported
```

## Why this is a fair benchmark

Comparing your number to a number from another paper compares datasets, splits,
augmentation and tuning effort as much as architectures. Here:

- every model sees **identical splits, augmentation, schedule and epoch budget**;
- group-aware splitting is enforced where the dataset publishes a lesion identifier
  (HAM10000 only — the others publish none, which is stated as a limitation);
- differences are **tested** (Friedman, then paired Wilcoxon with Holm correction), not
  eyeballed;
- confidence intervals use the Nadeau–Bengio correction for overlapping training sets.

## Running

1. Attach the dataset (each loader cell names the source).
2. **Settings → Internet → On** — needed for `timm` and pretrained weights.
3. **Accelerator → GPU T4** — mandatory; on CPU a single fold takes hours.
4. **Run All.**

Each writes `benchmark_row_*.json`. Gather all four next to any one notebook and the
final cell builds the cross-dataset table.

Reduce `FOLDS` or `EPOCHS` in the run cell if a session is too short. Both are stated in
the output so the budget is reported alongside the result.

## Reading the outcome

| `dermnet_rank` | Meaning |
|---|---|
| 1 on all four | strong: the architecture wins under an equal budget everywhere |
| 1 on some | dataset-dependent; report per dataset and discuss why |
| not 1 | the architecture does not lead on that data; the claim must match |

`msca_gain_pp` positive and significant on more than one dataset is what supports a claim
about the fusion block specifically.

A result where DERM-Net ranks first but does not beat its own no-MSCA ablation is still
publishable — it just makes a different claim: that dual-backbone pretrained fusion is
effective for this task, not that MSCA is.

## Caveats for any write-up

- Absolute numbers are not comparable to published leaderboard results, which use
  different splits, budgets and often no cross-validation.
- Only HAM10000 supports group-aware splitting; the others may be optimistic.
- One training budget is applied to every model. A larger budget could favour larger
  models. This is stated rather than tuned away, because per-model tuning on the same
  data is how benchmarks stop being fair.
