"""
POC experiment: Run GNN and LightGBM on the sampled citation ER subset.
Adapted from RelBench examples (gnn_link.py and lightgbm_link.py).
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

# Add RelBench to path
sys.path.insert(0, "/data/home/lubarsky/phd/er/relbench")

import numpy as np
import pandas as pd
import torch
import duckdb

from relbench.base import Database, Dataset, Table, TaskType, RecommendationTask
from relbench.metrics import link_prediction_precision, link_prediction_recall, link_prediction_map

# ============================================================
# 1. Define the POC Dataset (uses the small sampled CSVs)
# ============================================================

POC_PATH = "/data/home/lubarsky/phd/er_project/poc_data"

def enforce_referential_integrity(child_df, fk_col, parent_df, parent_pk_col):
    if fk_col not in child_df.columns:
        return child_df
    if parent_df.index.name != parent_pk_col:
        parent_df = parent_df.set_index(parent_pk_col)
    valid = child_df[fk_col].isin(parent_df.index)
    removed = (~valid).sum()
    if removed > 0:
        print(f"  Removed {removed} invalid FK rows ({fk_col})")
    return child_df[valid].reset_index(drop=True)


def replace_pubid_with_pubkey(df, columns, mapping):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping)
    return df


class POCCitationDataset(Dataset):
    name = "poc-citation"
    val_timestamp = pd.Timestamp("2008-01-01")
    test_timestamp = pd.Timestamp("2014-01-02")

    def make_db(self) -> Database:
        default_date = pd.Timestamp("1937-01-01")

        # Load tables
        dblp_pub = pd.read_csv(os.path.join(POC_PATH, "dblp_publication.csv"))
        dblp_article = pd.read_csv(os.path.join(POC_PATH, "dblp_article.csv"), low_memory=False)
        dblp_book = pd.read_csv(os.path.join(POC_PATH, "dblp_book.csv"), low_memory=False)
        dblp_incol = pd.read_csv(os.path.join(POC_PATH, "dblp_incollection.csv"), low_memory=False)
        dblp_inproc = pd.read_csv(os.path.join(POC_PATH, "dblp_inproceedings.csv"), low_memory=False)
        dblp_author = pd.read_csv(os.path.join(POC_PATH, "dblp_author.csv"))
        dblp_authored = pd.read_csv(os.path.join(POC_PATH, "dblp_authored.csv"))
        authors_exp = pd.read_csv(os.path.join(POC_PATH, "authors_expanded.csv"), low_memory=False)
        citations_exp = pd.read_csv(os.path.join(POC_PATH, "citations_expanded.csv"), low_memory=False)
        papers_exp = pd.read_csv(os.path.join(POC_PATH, "papers_expanded.csv"), low_memory=False)
        abstracts_exp = pd.read_csv(os.path.join(POC_PATH, "abstracts_expanded.csv"), low_memory=False)

        # Process dates
        dblp_pub["year"] = pd.to_datetime(dblp_pub["year"], format='%Y-%m-%d', errors='coerce').fillna(default_date)
        papers_exp["year"] = pd.to_datetime(papers_exp["year"], errors='coerce').fillna(default_date)

        # Create pubid -> pubkey mapping
        pubid_to_pubkey = pd.Series(
            dblp_pub["pubkey"].values,
            index=dblp_pub["pubid"].astype(str)
        ).to_dict()

        # Replace pubid with pubkey in child tables
        for df in [dblp_article, dblp_book, dblp_incol, dblp_inproc, dblp_authored]:
            replace_pubid_with_pubkey(df, ["pubid"], pubid_to_pubkey)

        papers_exp["externalid_dblp"] = papers_exp["externalid_dblp"].astype(str)
        papers_exp = papers_exp.rename(columns={"first_author_id": "authorId"})

        # Referential integrity
        print("Enforcing referential integrity...")
        dblp_article = enforce_referential_integrity(dblp_article, "pubid", dblp_pub, "pubkey")
        dblp_book = enforce_referential_integrity(dblp_book, "pubid", dblp_pub, "pubkey")
        dblp_incol = enforce_referential_integrity(dblp_incol, "pubid", dblp_pub, "pubkey")
        dblp_inproc = enforce_referential_integrity(dblp_inproc, "pubid", dblp_pub, "pubkey")
        dblp_authored = enforce_referential_integrity(dblp_authored, "pubid", dblp_pub, "pubkey")
        dblp_authored = enforce_referential_integrity(dblp_authored, "authorid", dblp_author, "authorid")
        abstracts_exp = enforce_referential_integrity(abstracts_exp, "corpusid", papers_exp, "corpusid")
        citations_exp = enforce_referential_integrity(citations_exp, "citingcorpusid", papers_exp, "corpusid")
        citations_exp = enforce_referential_integrity(citations_exp, "citedcorpusid", papers_exp, "corpusid")
        papers_exp = enforce_referential_integrity(papers_exp, "authorId", authors_exp, "authorid")
        # Note: we do NOT enforce RI on externalid_dblp because many papers legitimately
        # don't have a DBLP match (those are the "unknown" cases the ER model should learn about).
        # We only drop papers whose externalid_dblp is set but doesn't match any pubkey.
        has_dblp = papers_exp["externalid_dblp"].notna() & (papers_exp["externalid_dblp"] != "nan") & (papers_exp["externalid_dblp"] != "")
        valid_dblp = papers_exp["externalid_dblp"].isin(dblp_pub.set_index("pubkey").index)
        invalid_mask = has_dblp & ~valid_dblp
        print(f"  Removed {invalid_mask.sum()} papers with invalid DBLP refs (keeping {(~has_dblp).sum()} without DBLP)")
        papers_exp = papers_exp[~invalid_mask].reset_index(drop=True)

        return Database(
            table_dict={
                "dblp_publication": Table(df=dblp_pub, fkey_col_to_pkey_table={}, pkey_col="pubkey", time_col="year"),
                "dblp_article": Table(df=dblp_article, fkey_col_to_pkey_table={"pubid": "dblp_publication"}, pkey_col=None, time_col=None),
                "dblp_book": Table(df=dblp_book, fkey_col_to_pkey_table={"pubid": "dblp_publication"}, pkey_col=None, time_col=None),
                "dblp_incollection": Table(df=dblp_incol, fkey_col_to_pkey_table={"pubid": "dblp_publication"}, pkey_col=None, time_col=None),
                "dblp_inproceedings": Table(df=dblp_inproc, fkey_col_to_pkey_table={"pubid": "dblp_publication"}, pkey_col=None, time_col=None),
                "dblp_author": Table(df=dblp_author, fkey_col_to_pkey_table={}, pkey_col="authorid", time_col=None),
                "dblp_authored": Table(df=dblp_authored, fkey_col_to_pkey_table={"pubid": "dblp_publication", "authorid": "dblp_author"}, pkey_col=None, time_col=None),
                "authors_expanded": Table(df=authors_exp, fkey_col_to_pkey_table={}, pkey_col="authorid", time_col=None),
                "citations_expanded": Table(df=citations_exp, fkey_col_to_pkey_table={"citingcorpusid": "papers_expanded", "citedcorpusid": "papers_expanded"}, pkey_col=None, time_col=None),
                "papers_expanded": Table(df=papers_exp, fkey_col_to_pkey_table={"authorId": "authors_expanded", "externalid_dblp": "dblp_publication"}, pkey_col="corpusid", time_col="year"),
                "abstracts_expanded": Table(df=abstracts_exp, fkey_col_to_pkey_table={"corpusid": "papers_expanded"}, pkey_col=None, time_col=None),
            }
        )


class POCLinkageTask(RecommendationTask):
    name = "poc-citation-linkage"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "externalid_dblp"
    src_entity_table = "papers_expanded"
    dst_entity_col = "pubkey"
    dst_entity_table = "dblp_publication"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 * 6)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        dblp_publication = db.table_dict["dblp_publication"].df
        papers_expanded = db.table_dict["papers_expanded"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                papers_expanded.externalid_dblp,
                LIST(dblp_publication.pubkey) as pubkey
            FROM
                timestamp_df t
            LEFT JOIN
                dblp_publication
            ON
                dblp_publication.year > t.timestamp AND
                dblp_publication.year <= t.timestamp + INTERVAL '{self.timedelta.days} days'
            INNER JOIN
                papers_expanded
            ON
                dblp_publication.pubkey = papers_expanded.externalid_dblp
            GROUP BY
                t.timestamp,
                papers_expanded.externalid_dblp
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


# ============================================================
# 2. Run LightGBM Baseline
# ============================================================

def run_lightgbm_baseline(dataset, task):
    """Simple LightGBM baseline using paper features."""
    import lightgbm as lgb

    print("\n" + "="*60)
    print("Running LightGBM Baseline")
    print("="*60)

    db = dataset.get_db()
    papers = db.table_dict["papers_expanded"].df.copy()
    dblp_pubs = db.table_dict["dblp_publication"].df.copy()

    # Create simple features for papers
    feature_cols = ["citation_count", "reference_count", "influential_citation_count", "is_open_access"]
    for col in feature_cols:
        if col in papers.columns:
            papers[col] = pd.to_numeric(papers[col], errors="coerce").fillna(0)

    # Get task tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    results = {}
    for split_name, split_table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        if split_table is not None and len(split_table.df) > 0:
            results[split_name] = {
                "num_rows": len(split_table.df),
                "num_unique_src": split_table.df["externalid_dblp"].nunique() if "externalid_dblp" in split_table.df.columns else 0,
            }
            print(f"  {split_name}: {results[split_name]}")
        else:
            results[split_name] = {"num_rows": 0}
            print(f"  {split_name}: empty or None")

    print("\nLightGBM baseline: Dataset loaded and splits verified.")
    return results


# ============================================================
# 3. Run GNN (graph construction test)
# ============================================================

def run_gnn_test(dataset, task):
    """Test GNN graph construction on the POC dataset."""
    print("\n" + "="*60)
    print("Running GNN Graph Construction Test")
    print("="*60)

    db = dataset.get_db()

    try:
        from relbench.modeling.graph import make_pkey_fkey_graph
        from relbench.modeling.utils import get_stype_proposal
        import torch_frame

        # Get column type proposals
        col_to_stype_dict = get_stype_proposal(db)
        print(f"  Column types proposed for {len(col_to_stype_dict)} tables")

        # Remove text columns to avoid needing a text embedder
        for table_name in list(col_to_stype_dict.keys()):
            col_to_stype_dict[table_name] = {
                col: stype for col, stype in col_to_stype_dict[table_name].items()
                if stype != torch_frame.text_embedded and stype != torch_frame.text_tokenized
            }

        # Build the heterogeneous graph
        print("  Building heterogeneous graph...")
        t0 = time.time()
        data, col_stats_dict = make_pkey_fkey_graph(
            db,
            col_to_stype_dict=col_to_stype_dict,
        )
        t1 = time.time()

        print(f"  Graph built in {t1-t0:.1f}s")
        print(f"  Node types: {data.node_types}")
        print(f"  Edge types: {data.edge_types}")

        for ntype in data.node_types:
            n = data[ntype].num_nodes
            print(f"    {ntype}: {n} nodes")

        results = {
            "node_types": list(data.node_types),
            "edge_types": [str(et) for et in data.edge_types],
            "num_node_types": len(data.node_types),
            "num_edge_types": len(data.edge_types),
            "build_time_sec": round(t1 - t0, 2),
        }

        return results

    except Exception as e:
        print(f"  GNN graph construction failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
# 4. Main
# ============================================================

def main():
    print("="*60)
    print("POC Experiment: DBLP-Semantic Scholar ER Dataset")
    print("="*60)

    # Create dataset
    print("\nCreating POC dataset...")
    dataset = POCCitationDataset(cache_dir="/tmp/poc_cache")
    db = dataset.get_db()

    print("\nDatabase tables:")
    for name, table in db.table_dict.items():
        print(f"  {name}: {len(table.df)} rows, pk={table.pkey_col}, time={table.time_col}")

    # Create task
    task = POCLinkageTask(dataset)

    # Run LightGBM test
    lgb_results = run_lightgbm_baseline(dataset, task)

    # Run GNN test
    gnn_results = run_gnn_test(dataset, task)

    # Save results
    all_results = {
        "dataset": "DBLP-Semantic Scholar POC",
        "tables": {name: len(table.df) for name, table in db.table_dict.items()},
        "lgb_results": lgb_results,
        "gnn_results": gnn_results,
    }

    results_path = "/data/home/lubarsky/phd/er_project/poc_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print("\n" + "="*60)
    print("POC Experiment Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
