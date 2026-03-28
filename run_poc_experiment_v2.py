"""
POC experiment v2: Run GNN and LightGBM on the larger stratified subset.
Key fixes from v1:
  - Let RelBench handle PK/FK re-indexing (reindex_pkeys_and_fkeys)
  - Temporal splits: val=2019, test=2022, timedelta=3yr
  - Actual LightGBM training with positive/negative sampling
  - GNN graph construction + training
  - Reports precision, recall, MAP@12
"""
import os
import sys
import json
import time
import copy
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/data/home/lubarsky/phd/er/relbench")

import numpy as np
import pandas as pd
import torch
import torch_frame
from torch_frame import stype
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric

from relbench.base import Database, Dataset, Table, TaskType, RecommendationTask
from relbench.metrics import link_prediction_precision, link_prediction_recall, link_prediction_map
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

POC_PATH = "/data/home/lubarsky/phd/er_project/poc_data_v2"

# ============================================================
# 1. Dataset: load CSVs, let framework re-index PKs/FKs
# ============================================================

class POCCitationDatasetV2(Dataset):
    name = "poc-citation-v2"
    val_timestamp = pd.Timestamp("2019-01-01")
    test_timestamp = pd.Timestamp("2022-01-01")

    def make_db(self) -> Database:
        default_date = pd.Timestamp("1970-01-01")

        print("Loading tables...")
        dblp_pub = pd.read_csv(os.path.join(POC_PATH, "dblp_publication.csv"), low_memory=False)
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

        # Parse dates
        dblp_pub["year"] = pd.to_datetime(dblp_pub["year"], format='%Y-%m-%d', errors='coerce').fillna(default_date)
        papers_exp["year"] = pd.to_datetime(papers_exp["year"], errors='coerce').fillna(default_date)
        papers_exp = papers_exp.rename(columns={"first_author_id": "authorId"})

        # Deduplicate parent tables on their PKs
        dblp_pub = dblp_pub.drop_duplicates(subset="pubkey").reset_index(drop=True)
        dblp_author = dblp_author.drop_duplicates(subset="authorid").reset_index(drop=True)
        authors_exp = authors_exp.drop_duplicates(subset="authorid").reset_index(drop=True)
        papers_exp = papers_exp.drop_duplicates(subset="corpusid").reset_index(drop=True)

        # Convert pubid in child tables from int -> pubkey string
        # (so the FK column matches dblp_publication.pubkey)
        pubid_to_pubkey = pd.Series(
            dblp_pub["pubkey"].values,
            index=dblp_pub["pubid"].astype(str)
        ).to_dict()

        for df in [dblp_article, dblp_book, dblp_incol, dblp_inproc, dblp_authored]:
            if "pubid" in df.columns:
                df["pubid"] = df["pubid"].astype(str).map(pubid_to_pubkey)

        # Clean up externalid_dblp: ensure NaN for papers without DBLP match
        papers_exp["externalid_dblp"] = papers_exp["externalid_dblp"].astype(str)
        papers_exp.loc[
            papers_exp["externalid_dblp"].isin(["nan", "", "None", "NaN"]),
            "externalid_dblp"
        ] = np.nan

        # Enforce referential integrity
        valid_pubkeys = set(dblp_pub["pubkey"].dropna())
        valid_dblp_authorids = set(dblp_author["authorid"].dropna())
        valid_s2_authorids = set(authors_exp["authorid"].dropna())
        valid_corpusids = set(papers_exp["corpusid"].dropna())

        def filter_fk(df, col, valid_set, name):
            before = len(df)
            mask = df[col].isna() | df[col].isin(valid_set)
            df = df[mask].reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                print(f"  {name}.{col}: dropped {dropped} invalid FK rows")
            return df

        dblp_article = filter_fk(dblp_article, "pubid", valid_pubkeys, "dblp_article")
        dblp_book = filter_fk(dblp_book, "pubid", valid_pubkeys, "dblp_book")
        dblp_incol = filter_fk(dblp_incol, "pubid", valid_pubkeys, "dblp_incollection")
        dblp_inproc = filter_fk(dblp_inproc, "pubid", valid_pubkeys, "dblp_inproceedings")
        dblp_authored = filter_fk(dblp_authored, "pubid", valid_pubkeys, "dblp_authored")
        dblp_authored = filter_fk(dblp_authored, "authorid", valid_dblp_authorids, "dblp_authored")
        citations_exp = filter_fk(citations_exp, "citingcorpusid", valid_corpusids, "citations_expanded")
        citations_exp = filter_fk(citations_exp, "citedcorpusid", valid_corpusids, "citations_expanded")
        abstracts_exp = filter_fk(abstracts_exp, "corpusid", valid_corpusids, "abstracts_expanded")
        papers_exp = filter_fk(papers_exp, "authorId", valid_s2_authorids, "papers_expanded")
        # For externalid_dblp: only drop papers with an invalid (non-NaN) FK
        has_dblp = papers_exp["externalid_dblp"].notna()
        valid_dblp = papers_exp["externalid_dblp"].isin(valid_pubkeys)
        invalid_mask = has_dblp & ~valid_dblp
        print(f"  papers_expanded.externalid_dblp: dropped {invalid_mask.sum()} invalid, keeping {(~has_dblp).sum()} without DBLP")
        papers_exp = papers_exp[~invalid_mask].reset_index(drop=True)

        # Re-validate corpusid references after papers_exp changes
        valid_corpusids = set(papers_exp["corpusid"].dropna())
        citations_exp = filter_fk(citations_exp, "citingcorpusid", valid_corpusids, "citations_expanded(2)")
        citations_exp = filter_fk(citations_exp, "citedcorpusid", valid_corpusids, "citations_expanded(2)")
        abstracts_exp = filter_fk(abstracts_exp, "corpusid", valid_corpusids, "abstracts_expanded(2)")

        print(f"\nTable sizes before re-indexing:")
        for name, df in [("dblp_publication", dblp_pub), ("dblp_article", dblp_article),
                         ("dblp_book", dblp_book), ("dblp_incollection", dblp_incol),
                         ("dblp_inproceedings", dblp_inproc), ("dblp_author", dblp_author),
                         ("dblp_authored", dblp_authored), ("authors_expanded", authors_exp),
                         ("citations_expanded", citations_exp), ("papers_expanded", papers_exp),
                         ("abstracts_expanded", abstracts_exp)]:
            print(f"  {name}: {len(df)}")

        has_match = papers_exp["externalid_dblp"].notna()
        print(f"\nPapers with DBLP match: {has_match.sum()} / {len(papers_exp)}")
        print(f"DBLP pub year range: {dblp_pub['year'].min()} to {dblp_pub['year'].max()}")
        print(f"Papers year range: {papers_exp['year'].min()} to {papers_exp['year'].max()}")

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


# ============================================================
# 2. Task definition
# ============================================================

class POCLinkageTaskV2(RecommendationTask):
    """Entity resolution as link prediction: for each S2 paper, predict matching DBLP publication."""
    name = "poc-linkage-v2"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "corpusid"
    src_entity_table = "papers_expanded"
    dst_entity_col = "pubkey"
    dst_entity_table = "dblp_publication"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 * 3)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        papers = db.table_dict["papers_expanded"].df
        dblp_pub = db.table_dict["dblp_publication"].df

        rows = []
        for ts in timestamps:
            end_ts = ts + self.timedelta
            # DBLP publications in this time window
            dblp_in_window = set(
                dblp_pub[(dblp_pub["year"] > ts) & (dblp_pub["year"] <= end_ts)]["pubkey"].values
            )
            if not dblp_in_window:
                continue

            # Papers with valid DBLP match pointing to a pub in this window
            matched = papers[
                papers["externalid_dblp"].notna() &
                papers["externalid_dblp"].isin(dblp_in_window)
            ]

            for _, row in matched.iterrows():
                rows.append({
                    "timestamp": ts,
                    "corpusid": int(row["corpusid"]),
                    "pubkey": [int(row["externalid_dblp"])],
                })

        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=["timestamp", "corpusid", "pubkey"])

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
# 3. LightGBM baseline
# ============================================================

def run_lightgbm(dataset, task):
    """LightGBM baseline: binary classification of (paper, dblp_pub) pairs."""
    print("\n" + "="*60)
    print("LightGBM Baseline")
    print("="*60)

    db = dataset.get_db()
    src_df = db.table_dict[task.src_entity_table].df.copy()
    dst_df = db.table_dict[task.dst_entity_table].df.copy()
    src_table = db.table_dict[task.src_entity_table]
    dst_table = db.table_dict[task.dst_entity_table]

    # Get stype proposals, remove text and multicategorical columns
    col_to_stype_dict = get_stype_proposal(db)
    for table_name in list(col_to_stype_dict.keys()):
        col_to_stype_dict[table_name] = {
            col: st for col, st in col_to_stype_dict[table_name].items()
            if st not in (torch_frame.text_embedded, torch_frame.text_tokenized,
                          torch_frame.multicategorical)
        }

    # Get task tables
    print("\nGetting task tables...")
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    for split_name, t in [("train", train_table), ("val", val_table), ("test", test_table)]:
        n = len(t.df) if t is not None else 0
        print(f"  {split_name}: {n} rows")

    if len(train_table.df) == 0:
        print("ERROR: Empty training set!")
        return {"error": "empty training set"}

    # Prepare col_to_stype for the merged dataframe
    src_col_to_stype = copy.deepcopy(col_to_stype_dict.get(task.src_entity_table, {}))
    dst_col_to_stype = copy.deepcopy(col_to_stype_dict.get(task.dst_entity_table, {}))
    remove_pkey_fkey(src_col_to_stype, src_table)
    remove_pkey_fkey(dst_col_to_stype, dst_table)

    # Handle column name collisions between src and dst
    overlap_cols = set(src_col_to_stype.keys()) & set(dst_col_to_stype.keys())
    for col in overlap_cols:
        src_col_to_stype[f"{col}_x"] = src_col_to_stype.pop(col)
        dst_col_to_stype[f"{col}_y"] = dst_col_to_stype.pop(col)

    col_to_stype = {}
    col_to_stype.update(src_col_to_stype)
    col_to_stype.update(dst_col_to_stype)
    TARGET_COL = "link_target"
    col_to_stype[TARGET_COL] = torch_frame.categorical

    src_entity = task.src_entity_col
    dst_entity = task.dst_entity_col

    def create_training_df(table, src_df, dst_df, neg_ratio=1):
        """Create positive + negative pairs for binary classification."""
        df = table.df.copy()
        df = df.explode(dst_entity)
        df[TARGET_COL] = 1

        # Join with src entity features
        df = df.merge(src_df, how="left", left_on=src_entity, right_on=src_table.pkey_col)

        # Create negative samples
        n_neg = int(len(df) * neg_ratio)
        neg_df = df.drop(columns=[dst_entity, TARGET_COL]).copy()
        neg_df = neg_df.sample(n=n_neg, replace=True, random_state=SEED).reset_index(drop=True)
        neg_df[dst_entity] = np.random.choice(dst_df[dst_table.pkey_col].values, size=n_neg)
        neg_df[TARGET_COL] = 0

        df = pd.concat([df, neg_df], ignore_index=True)
        df = pd.merge(df, dst_df, how="left", left_on=dst_entity, right_on=dst_table.pkey_col)
        return df

    print("\nPreparing training data...")
    train_df = create_training_df(train_table, src_df, dst_df)
    val_df = create_training_df(val_table, src_df, dst_df) if len(val_table.df) > 0 else None

    print(f"  Train pairs: {len(train_df)} ({(train_df[TARGET_COL]==1).sum()} pos, {(train_df[TARGET_COL]==0).sum()} neg)")
    if val_df is not None:
        print(f"  Val pairs: {len(val_df)} ({(val_df[TARGET_COL]==1).sum()} pos, {(val_df[TARGET_COL]==0).sum()} neg)")

    # Materialize torch_frame datasets
    print("\nMaterializing features...")
    try:
        train_dataset = torch_frame.data.Dataset(
            df=train_df, col_to_stype=col_to_stype, target_col=TARGET_COL,
        ).materialize()
    except Exception as e:
        print(f"  Materialization failed: {e}")
        print("  Retrying with numeric-only features...")
        col_to_stype = {
            col: st for col, st in col_to_stype.items()
            if st in (torch_frame.numerical, torch_frame.categorical)
        }
        col_to_stype[TARGET_COL] = torch_frame.categorical
        train_dataset = torch_frame.data.Dataset(
            df=train_df, col_to_stype=col_to_stype, target_col=TARGET_COL,
        ).materialize()

    tf_train = train_dataset.tensor_frame
    tf_val = train_dataset.convert_to_tensor_frame(val_df) if val_df is not None else tf_train

    # Train LightGBM
    print("\nTraining LightGBM...")
    model = LightGBM(task_type=train_dataset.task_type, metric=Metric.ROCAUC)
    model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=10)
    print("  Training complete!")

    # Evaluate on val and test using link prediction metrics
    from collections import Counter
    results = {}

    for split_name, split_table in [("val", val_table), ("test", test_table)]:
        if split_table is None or len(split_table.df) == 0:
            results[split_name] = {"error": "empty split"}
            continue

        eval_df = split_table.df.copy()
        eval_input = eval_df.drop(columns=[dst_entity])

        # Candidate dst entities: popular from train + random
        all_train_dsts = [d for dsts in train_table.df[dst_entity] for d in dsts]
        dst_counter = Counter(all_train_dsts)
        top_dsts = [d for d, _ in dst_counter.most_common(task.eval_k * 2)]
        random_dsts = list(np.random.choice(dst_df[dst_table.pkey_col].values, size=task.eval_k * 2))
        candidate_dsts = list(set(top_dsts + random_dsts))[:task.eval_k * 3]

        pred_indices = []
        for _, row in eval_input.iterrows():
            src_id = row[src_entity]
            cand_df = pd.DataFrame({
                src_entity: [src_id] * len(candidate_dsts),
                dst_entity: candidate_dsts,
                TARGET_COL: [0] * len(candidate_dsts),
                "timestamp": [row["timestamp"]] * len(candidate_dsts),
            })
            cand_df = cand_df.merge(src_df, how="left", left_on=src_entity, right_on=src_table.pkey_col)
            cand_df = cand_df.merge(dst_df, how="left", left_on=dst_entity, right_on=dst_table.pkey_col)

            try:
                tf_cand = train_dataset.convert_to_tensor_frame(cand_df)
                scores = model.predict(tf_test=tf_cand).numpy()
                topk_idx = np.argsort(-scores)[:task.eval_k]
                pred = [candidate_dsts[i] for i in topk_idx]
            except Exception:
                pred = candidate_dsts[:task.eval_k]

            while len(pred) < task.eval_k:
                pred.append(-1)
            pred_indices.append(pred[:task.eval_k])

        pred_array = np.array(pred_indices, dtype=int)
        metrics = task.evaluate(pred_array, split_table)
        results[split_name] = metrics
        print(f"\n  {split_name} metrics: {metrics}")

    return results


# ============================================================
# 4. GNN experiment
# ============================================================

def run_gnn(dataset, task):
    """GNN: build graph, train with BPR loss, evaluate link prediction."""
    print("\n" + "="*60)
    print("GNN Experiment")
    print("="*60)

    db = dataset.get_db()

    # Get stype proposals, remove text and multicategorical columns
    col_to_stype_dict = get_stype_proposal(db)
    for table_name in list(col_to_stype_dict.keys()):
        col_to_stype_dict[table_name] = {
            col: st for col, st in col_to_stype_dict[table_name].items()
            if st not in (torch_frame.text_embedded, torch_frame.text_tokenized,
                          torch_frame.multicategorical)
        }

    print("\nBuilding heterogeneous graph...")
    t0 = time.time()
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
    )
    t1 = time.time()
    print(f"  Graph built in {t1-t0:.1f}s")
    print(f"  Node types: {len(data.node_types)}, Edge types: {len(data.edge_types)}")
    for ntype in data.node_types:
        print(f"    {ntype}: {data[ntype].num_nodes} nodes")

    # Get task tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    print(f"\n  Train: {len(train_table.df)} rows")
    print(f"  Val: {len(val_table.df)} rows")
    print(f"  Test: {len(test_table.df)} rows")

    # Try GNN training
    try:
        from relbench.modeling.graph import get_link_train_table_input
        from relbench.modeling.loader import LinkNeighborLoader
        from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
        from torch_geometric.loader import NeighborLoader
        from torch_geometric.nn import MLP
        from torch.nn import Embedding, ModuleDict
        import torch.nn.functional as F

        device = torch.device("cpu")
        channels = 64
        num_layers = 2
        batch_size = 256
        epochs = 10
        lr = 0.001
        num_neighbors = [64, 32]

        class GNNModel(torch.nn.Module):
            def __init__(self, data, col_stats_dict, channels, num_layers):
                super().__init__()
                self.encoder = HeteroEncoder(
                    channels=channels,
                    node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
                    node_to_col_stats=col_stats_dict,
                )
                self.temporal_encoder = HeteroTemporalEncoder(
                    node_types=[nt for nt in data.node_types if "time" in data[nt]],
                    channels=channels,
                )
                self.gnn = HeteroGraphSAGE(
                    node_types=data.node_types, edge_types=data.edge_types,
                    channels=channels, aggr="sum", num_layers=num_layers,
                )
                self.head = MLP(channels, out_channels=channels, norm="layer_norm", num_layers=1)
                self.embedding_dict = ModuleDict({
                    task.dst_entity_table: Embedding(data.num_nodes_dict[task.dst_entity_table], channels)
                })

            def forward(self, batch, entity_table):
                seed_time = batch[entity_table].seed_time
                x_dict = self.encoder(batch.tf_dict)
                rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
                for nt, rt in rel_time_dict.items():
                    x_dict[nt] = x_dict[nt] + rt
                for nt, emb in self.embedding_dict.items():
                    x_dict[nt] = x_dict[nt] + emb(batch[nt].n_id)
                x_dict = self.gnn(
                    x_dict, batch.edge_index_dict,
                    batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict,
                )
                return self.head(x_dict[entity_table][:seed_time.size(0)])

        train_table_input = get_link_train_table_input(train_table, task)
        train_loader = LinkNeighborLoader(
            data=data, num_neighbors=num_neighbors, time_attr="time",
            src_nodes=train_table_input.src_nodes,
            dst_nodes=train_table_input.dst_nodes,
            num_dst_nodes=train_table_input.num_dst_nodes,
            src_time=train_table_input.src_time,
            share_same_time=True, batch_size=batch_size,
            shuffle=False, num_workers=0,
        )

        model = GNNModel(data, col_stats_dict, channels, num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print("\nTraining GNN...")
        best_val_metric = 0
        best_state = None
        max_steps = 200

        for epoch in range(1, epochs + 1):
            model.train()
            loss_accum = count = steps = 0

            for batch in train_loader:
                src_batch, batch_pos_dst, batch_neg_dst = batch
                src_batch = src_batch.to(device)
                batch_pos_dst = batch_pos_dst.to(device)
                batch_neg_dst = batch_neg_dst.to(device)

                x_src = model(src_batch, task.src_entity_table)
                x_pos = model(batch_pos_dst, task.dst_entity_table)
                x_neg = model(batch_neg_dst, task.dst_entity_table)

                pos_score = (x_src * x_pos).sum(dim=1).view(-1, 1)
                neg_score = x_src @ x_neg.t()
                loss = F.softplus(neg_score - pos_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accum += float(loss) * x_src.size(0)
                count += x_src.size(0)
                steps += 1
                if steps >= max_steps:
                    break

            avg_loss = loss_accum / count if count > 0 else float("nan")
            print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}")

            # Val evaluation every 2 epochs
            if epoch % 2 == 0 and len(val_table.df) > 0:
                model.eval()
                with torch.no_grad():
                    seed_time = int(dataset.val_timestamp.timestamp())
                    dst_loader = NeighborLoader(
                        data, num_neighbors=num_neighbors, time_attr="time",
                        input_nodes=task.dst_entity_table,
                        input_time=torch.full((task.num_dst_nodes,), seed_time, dtype=torch.long),
                        batch_size=batch_size, shuffle=False, num_workers=0,
                    )
                    dst_embs = []
                    for b in dst_loader:
                        b = b.to(device)
                        dst_embs.append(model(b, task.dst_entity_table).detach())
                    dst_emb = torch.cat(dst_embs, dim=0)

                    src_indices = torch.from_numpy(val_table.df[task.src_entity_col].values)
                    src_loader = NeighborLoader(
                        data, num_neighbors=num_neighbors, time_attr="time",
                        input_nodes=(task.src_entity_table, src_indices),
                        input_time=torch.full((len(src_indices),), seed_time, dtype=torch.long),
                        batch_size=batch_size, shuffle=False, num_workers=0,
                    )
                    preds = []
                    for b in src_loader:
                        b = b.to(device)
                        emb = model(b, task.src_entity_table)
                        _, topk = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
                        preds.append(topk.cpu())

                    pred = torch.cat(preds, dim=0).numpy()
                    val_metrics = task.evaluate(pred, val_table)
                    print(f"    Val: {val_metrics}")
                    if val_metrics.get("link_prediction_map", 0) >= best_val_metric:
                        best_val_metric = val_metrics["link_prediction_map"]
                        best_state = copy.deepcopy(model.state_dict())

        # Final evaluation
        results = {"training_completed": True, "epochs": epochs}
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            for split_name, split_table, ts in [
                ("val", val_table, dataset.val_timestamp),
                ("test", test_table, dataset.test_timestamp),
            ]:
                if split_table is None or len(split_table.df) == 0:
                    results[split_name] = {"error": "empty"}
                    continue

                seed_time = int(ts.timestamp())
                dst_loader = NeighborLoader(
                    data, num_neighbors=num_neighbors, time_attr="time",
                    input_nodes=task.dst_entity_table,
                    input_time=torch.full((task.num_dst_nodes,), seed_time, dtype=torch.long),
                    batch_size=batch_size, shuffle=False, num_workers=0,
                )
                dst_embs = []
                for b in dst_loader:
                    b = b.to(device)
                    dst_embs.append(model(b, task.dst_entity_table).detach())
                dst_emb = torch.cat(dst_embs, dim=0)

                src_indices = torch.from_numpy(split_table.df[task.src_entity_col].values)
                src_loader = NeighborLoader(
                    data, num_neighbors=num_neighbors, time_attr="time",
                    input_nodes=(task.src_entity_table, src_indices),
                    input_time=torch.full((len(src_indices),), seed_time, dtype=torch.long),
                    batch_size=batch_size, shuffle=False, num_workers=0,
                )
                preds = []
                for b in src_loader:
                    b = b.to(device)
                    emb = model(b, task.src_entity_table)
                    _, topk = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
                    preds.append(topk.cpu())

                pred = torch.cat(preds, dim=0).numpy()
                metrics = task.evaluate(pred, split_table)
                results[split_name] = metrics
                print(f"\n  Best {split_name}: {metrics}")

        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "graph_built": True,
            "num_node_types": len(data.node_types),
            "num_edge_types": len(data.edge_types),
            "training_error": str(e),
        }


# ============================================================
# 5. Main
# ============================================================

def main():
    print("="*60)
    print("POC Experiment v2: DBLP-Semantic Scholar ER")
    print("="*60)

    dataset = POCCitationDatasetV2(cache_dir="/tmp/poc_cache_v2")
    db = dataset.get_db()

    print("\nDatabase tables (after upto filter):")
    for name, table in db.table_dict.items():
        print(f"  {name}: {len(table.df)} rows, pk={table.pkey_col}, time={table.time_col}")
    print(f"  min_timestamp: {db.min_timestamp}")
    print(f"  max_timestamp: {db.max_timestamp}")

    # Also check the full db (without upto filter)
    full_db = dataset.get_db(upto_test_timestamp=False)
    print(f"\nFull DB max_timestamp: {full_db.max_timestamp}")
    print(f"Full DB dblp_pub rows: {len(full_db.table_dict['dblp_publication'].df)}")
    print(f"Full DB papers rows: {len(full_db.table_dict['papers_expanded'].df)}")

    task = POCLinkageTaskV2(dataset)

    # Run LightGBM
    lgb_results = run_lightgbm(dataset, task)

    # Run GNN
    gnn_results = run_gnn(dataset, task)

    # Save results
    all_results = {
        "dataset": "DBLP-Semantic Scholar POC v2",
        "tables": {name: len(table.df) for name, table in db.table_dict.items()},
        "val_timestamp": str(dataset.val_timestamp),
        "test_timestamp": str(dataset.test_timestamp),
        "lgb_results": lgb_results,
        "gnn_results": gnn_results,
    }

    results_path = "/data/home/lubarsky/phd/er_project/poc_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
