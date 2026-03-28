import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    rmse,
    roc_auc,
)


class LinkageTask(RecommendationTask):
    # TODO: rewrite this task
    r"""Predict the list of articles each customer will purchase in the next
    seven days"""

    name = "citation-cs-linkage"
    task_type = TaskType.LINK_PREDICTION
    # src_entity_col = "pubkey"
    # src_entity_table = "dblp_publication"
    # dst_entity_col = "externalid_dblp"
    # dst_entity_table = "papers_expanded"
    src_entity_col = "externalid_dblp"
    src_entity_table = "papers_expanded"
    dst_entity_col = "pubkey"
    dst_entity_table = "dblp_publication"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365*6)
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
                --dblp_publication.pubkey,
                -- We want here a list?
                --LIST(papers_expanded.externalid_dblp) as externalid_dblp  -- LIST(dblp_publication.pubkey) as externalid_dblp  -- LIST(DISTINCT ssd_papers.corpusid) AS corpusid
                papers_expanded.externalid_dblp,
                LIST(dblp_publication.pubkey) as pubkey
            FROM
                timestamp_df t
            LEFT JOIN
                dblp_publication
            ON
                dblp_publication.year > t.timestamp AND
                dblp_publication.year <= t.timestamp + INTERVAL '{self.timedelta} days'
            -- do i need a join with ssd_papers.corpusid?
            INNER JOIN
                papers_expanded
            ON
                dblp_publication.pubkey = papers_expanded.externalid_dblp
            GROUP BY
                t.timestamp,
                papers_expanded.externalid_dblp
                --dblp_publication.pubkey
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
