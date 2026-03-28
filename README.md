# Entity Resolution Dataset for RelBench

Project report and code for constructing a large-scale, multi-table Entity Resolution dataset from DBLP and Semantic Scholar, integrated into the [RelBench](https://relbench.stanford.edu/) framework.


## What this is

Most ER benchmarks are small two-table datasets. This project builds a larger one by combining:
- **DBLP** -- 6.9M computer science publications (XML dump → relational tables)
- **Semantic Scholar** -- 3.2M papers with citations, authors, and abstracts (bulk API → PostgreSQL)

Ground truth comes from DBLP keys embedded in Semantic Scholar's `externalIds` field, giving ~280K labeled matching pairs with no manual annotation.

The dataset is packaged as a custom RelBench `Dataset` and `RecommendationTask` (link prediction), with 11 relational tables and temporal train/val/test splits (val: 2019, test: 2022).

## Repo structure

```
report/          LaTeX source and compiled PDF of the project report
src/
  dblp/          DBLP XML parsing scripts
  semantic_scholar/  S2 API download and CSV conversion scripts
  sql/           PostgreSQL schema and staging table SQL
  relbench/      Custom RelBench Dataset and Task classes
notebooks/       Exploration and export notebooks
create_poc_subset_v2.py   Stratified sampling script for the POC subset
run_poc_experiment_v2.py  LightGBM + GNN experiments on the POC subset
poc_results_v2.json       Experiment results
```

## POC experiment results

Validation set (1,152 pairs, 23K candidates, no text features):

| Model | Precision@12 | Recall@12 | MAP@12 |
|---|---|---|---|
| LightGBM | 7.2e-5 | 8.7e-4 | 1.4e-4 |
| GNN (HeteroGraphSAGE) | 6.5e-4 | 7.8e-3 | 3.1e-3 |

The GNN is ~20× better than LightGBM without any text features, showing that graph structure (citations, co-authorship, cross-corpus links) provides useful signal for entity matching.

## Requirements

```bash
pip install relbench torch torch_geometric torch_frame lightgbm
```

The full dataset requires a PostgreSQL instance with the raw DBLP and Semantic Scholar data loaded. The POC scripts (`create_poc_subset_v2.py`, `run_poc_experiment_v2.py`) work on sampled CSVs and do not require the full database.
