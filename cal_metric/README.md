# `cal_metric` — test-metric summarization utilities

Two small, read-only utilities that aggregate the per-experiment `test_metrics.csv`
files (written by the test entrypoints) into a single Excel workbook. They are the
tools used to assemble the quantitative benchmark tables in the RAM-H1200 paper.

| Script | Task family | Reads (test CSV from) | Paper table(s) |
|---|---|---|---|
| `summarize_test_metrics.py` | **Segmentation** — hand bone segmentation & bone-erosion (BE) segmentation | `main_seg.py` / `main_be_seg.py --mode test` | **Table 2** (bone seg), **Table 3** (BE seg) |
| `summarize_score_metrics.py` | **Scoring** — SvdH BE & JSN score classification | `main_score_cls.py --mode test` | **Table 4** (BE scoring), **Table 5** (JSN scoring) |

> These scripts reproduce the **accuracy / quality-metric columns** of the tables.
> The `#P (M)` (parameter count) and `Time (ms)` (inference latency) columns are **not**
> produced here — they are separate model-profile measurements and are not stored in
> `test_metrics.csv`. Table 1 (dataset comparison) is descriptive and not computed here.

---

## `summarize_test_metrics.py` — segmentation (Tables 2 & 3)

**Input.** Each experiment directory holds a per-case `test_metrics.csv` whose first
column is `Case` (one row per test image, plus an optional final `average` row),
followed by aggregate `Mean *` columns and per-class columns, e.g.:

```
Case, DSC SvdH-BE-90, Mean DSC, NSD SvdH-BE-90, Mean NSD, VOE ..., Mean VOE, MSD ..., Mean MSD,
      Mean Overlap DSC, Mean Overlap NSD, Mean Precision, Mean Recall, ...
```

**What it computes.** For every `Mean *` column and every per-class column it reports
**mean ± sample standard deviation** across cases (the mean uses the CSV's `average`
row when present, otherwise the per-case mean; the SD is always computed from the
per-case values, `ddof = 1`).

**Output.** An `.xlsx` with two sheets: `Mean` (one row per experiment, `<metric> Mean`
and `<metric> SD` columns) and `PerClass` (same, per class). Format matches the paper's
`mean ± sd` style.

**Column mapping to the paper (Tables 2/3):**

| Paper column | CSV column |
|---|---|
| DSC | `Mean DSC` |
| NSD | `Mean NSD` |
| DSCO / NSDO (adjacent-bone overlap, Table 2) | `Mean Overlap DSC` / `Mean Overlap NSD` |
| VOE | `Mean VOE` |
| MSD | `Mean MSD` |
| REC / PREC (Table 3) | `Mean Recall` / `Mean Precision` |
| #P (M), Time (ms) | *not in CSV — measured separately* |

---

## `summarize_score_metrics.py` — scoring (Tables 4 & 5)

**Input.** Each experiment directory holds a `test_metrics.csv` with a `Scope` column
(`Overall` or `Joint`) and a `Name` column, followed by classification-metric columns:

```
Scope, Name, Accuracy, Precision, Recall, F1score, Specificity, BalancedAccuracy, DOR,
       QWK, MAE, Within1, Pos/Neg ACC, Binary Sensitivity, Binary Specificity
```

**What it computes.** It extracts the `Overall` row (dataset-level metrics) and all
`Joint` rows (per-joint metrics) and copies the values directly (scoring metrics are
reported as plain means, no SD).

**Output.** An `.xlsx` with two sheets: `Overall` (one row per experiment) and `Joint`
(one row per experiment, columns expanded as `<joint> <metric>`).

**Column mapping to the paper (Tables 4/5):**

| Paper column | CSV column |
|---|---|
| QWK | `QWK` |
| MAE | `MAE` |
| ACC (%) | `Accuracy` |
| BACC (%) | `BalancedAccuracy` |
| W1-ACC (%) | `Within1` |
| P/N-ACC (%) | `Pos/Neg ACC` |
| P/N-SEN (%) | `Binary Sensitivity` |
| #P (M), Time (ms) | *not in CSV — measured separately* |

---

## Usage

1. Edit the `EXP_PATHS` list at the top of the script to point at your experiment
   directories (each must contain a `test_metrics.csv`); a direct path to a CSV file
   also works.
2. Optionally set `OUTPUT_PATH` (destination `.xlsx`) and `DECIMALS` (rounding).
3. Run:

```bash
# Segmentation summary (Tables 2 & 3)
python cal_metric/summarize_test_metrics.py

# Scoring summary (Tables 4 & 5)
python cal_metric/summarize_score_metrics.py
```

The script writes the workbook to `OUTPUT_PATH` and prints a compact per-experiment
summary to the console.

## Requirements

- Python 3.8+
- `openpyxl` (Excel writing)

```bash
pip install openpyxl
```
