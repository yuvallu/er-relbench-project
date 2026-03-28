import os

import numpy as np
import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor


class CitationDataset(Dataset):
    name = "rel-citation"
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )
    # Train for the most recent 1 year out of 2 years of the original
    # time period
    train_start_timestamp = pd.Timestamp("1937-01-01")  # TODO: replace after testing pd.Timestamp("1937-01-01") --pd.Timestamp("2020-01-01")
    val_timestamp = pd.Timestamp("2008-01-01")  # TODO: why the val is only one "dimedelta" and not more
    test_timestamp = pd.Timestamp("2019-01-01")
    max_eval_time_frames = 1
    # task_cls_list = [LinkageTask]

    def make_db(self) -> Database:
        path = os.path.join("data", "citation-cs-linkage")
        zip = os.path.join(path, "citation-cs-linkage.zip")
        # TODO: name the tables.
        base_path = "/data/home/lubarsky/phd/s2ag-corpus/base_dir/csvs_dataset"
        dblp_publication = os.path.join(base_path, "dblp_publication.csv")
        dblp_article = os.path.join(base_path, "dblp_article.csv")
        dblp_book = os.path.join(base_path, "dblp_book.csv")
        dblp_incollection = os.path.join(base_path, "dblp_incollection.csv")
        dblp_inproceedings = os.path.join(base_path, "dblp_inproceedings.csv")
        dblp_author = os.path.join(base_path, "dblp_author.csv")
        dblp_authored = os.path.join(base_path, "dblp_authored.csv")

        abstracts_expanded = os.path.join(base_path, "abstracts_expanded.csv")
        authors_expanded = os.path.join(base_path, "authors_expanded.csv")
        citations_expanded = os.path.join(base_path, "citations_expanded.csv")
        paperids_expanded = os.path.join(base_path, "paperids_expanded.csv")
        papers_expanded = os.path.join(base_path, "papers_expanded.csv")
        # this code is needed for a later stage when I will store the data in a zip file
        # if not os.path.exists(customers):
        #     if not os.path.exists(zip):
        #         raise RuntimeError(
        #             f"Dataset not found. Please download "
        #             f"h-and-m-personalized-fashion-recommendations.zip from "
        #             f"'{self.url}' and move it to '{path}'. Once you have your"
        #             f"Kaggle API key, you can use the following command: "
        #             f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations"
        #         )
        #     else:
        #         print("Unpacking")
        #         shutil.unpack_archive(zip, Path(zip).parent)
        dblp_publication_df = pd.read_csv(dblp_publication)
        dblp_article_df = pd.read_csv(dblp_article)
        dblp_book_df = pd.read_csv(dblp_book)
        dblp_incollection_df = pd.read_csv(dblp_incollection)
        dblp_inproceedings_df = pd.read_csv(dblp_inproceedings, low_memory=False)
        dblp_author_df = pd.read_csv(dblp_author)
        dblp_authored_df = pd.read_csv(dblp_authored)

        ssd_papers_df = pd.read_csv(ssd_papers)


        default_date = self.train_start_timestamp

        # pre-processing the dataset
        ssd_papers_df["publicationdate"] = pd.to_datetime(ssd_papers_df["publicationdate"], format='%Y-%m-%d').fillna(default_date)
        dblp_publication_df["year"] = pd.to_datetime(dblp_publication_df["year"], format='%Y-%m-%d').fillna(default_date)

        # added another key_ column
        dblp_publication_df["key_pubkey"] = dblp_publication_df["pubkey"]
        dblp_article_df["key_pubid"] = dblp_article_df["pubid"]
        dblp_book_df["key_pubid"] = dblp_book_df["pubid"]
        dblp_incollection_df["key_pubid"] = dblp_incollection_df["pubid"]
        dblp_inproceedings_df["key_pubid"] = dblp_inproceedings_df["pubid"]
        dblp_author_df["key_authorid"] = dblp_author_df["authorid"]
        #
        ssd_papers_df["key_dblp_external_id"] = ssd_papers_df["dblp_external_id"]

        return Database(
            table_dict={
                "dblp_publication": Table(
                    df=dblp_publication_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="key_pubkey",  # this is the key with semantic scholar, but the real pkey is pubid
                    time_col="year",
                ),
                "dblp_article": Table(
                    df=dblp_article_df,
                    fkey_col_to_pkey_table={"key_pubkey": "dblp_publication"},
                    pkey_col="key_pubid",
                    time_col=None,
                ),
                "dblp_book": Table(
                    df=dblp_book_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col="key_pubid",
                    time_col=None,
                ),
                "dblp_incollection": Table(
                    df=dblp_incollection_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col="key_pubid",
                    time_col=None,
                ),
                "dblp_inproceedings": Table(
                    df=dblp_inproceedings_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col="key_pubid",
                    time_col=None,
                ),
                "dblp_author": Table(
                    df=dblp_author_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="key_authorid",
                    time_col=None,
                ),
                "dblp_authored": Table(
                    df=dblp_authored_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication", "authorid": "dblp_author"},
                    pkey_col=None,
                    time_col=None,
                ),


                "ssd_papers": Table(
                    df=ssd_papers_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="key_dblp_external_id",
                    time_col="publicationdate",
                ),
            }
        )
