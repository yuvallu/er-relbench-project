import os

import numpy as np
import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor

def replace_pubid_with_pubkey(df, columns, mapping):
    """
    Replace values in specified columns from pubid to pubkey using the provided mapping.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - columns (list of str): List of column names in df to replace.
    - mapping (dict): Dictionary mapping pubid (as string) to pubkey.

    Returns:
    - pd.DataFrame: The modified DataFrame with replaced values.
    """
    for col in columns:
        if col in df.columns:
            # Replace pubid with pubkey, handle missing mappings
            df[col] = df[col].astype(str).map(mapping)
            # Optionally, handle unmapped pubids
            if df[col].isnull().any():
                print(f"Warning: Some pubid values in column '{col}' were not found in the mapping and are set to NaN.")
    return df

def enforce_referential_integrity(child_df, fk_col, parent_df, parent_pk_col, child_table_name, parent_table_name):
    """
    Enforce referential integrity by removing rows in child_df where fk_col values
    do not exist in parent_df[parent_pk_col].

    Parameters:
    - child_df (pd.DataFrame): The child DataFrame.
    - fk_col (str): The foreign key column in child_df.
    - parent_df (pd.DataFrame): The parent DataFrame.
    - parent_pk_col (str): The primary key column in parent_df.
    - child_table_name (str): Name of the child table.
    - parent_table_name (str): Name of the parent table.

    Returns:
    - pd.DataFrame: The child DataFrame with invalid rows removed.
    """
    if fk_col not in child_df.columns:
        print(f"Foreign key column '{fk_col}' not found in '{child_table_name}'. Skipping this FK.")
        return child_df

    # Ensure parent_pk_col is the index for faster lookup
    if parent_df.index.name != parent_pk_col:
        parent_df = parent_df.set_index(parent_pk_col)

    # Identify valid foreign key values
    valid_fks = child_df[fk_col].isin(parent_df.index)
    invalid_fks = ~valid_fks

    num_invalid = invalid_fks.sum()
    if num_invalid > 0:
        print(f"Found {num_invalid} invalid foreign key(s) in '{child_table_name}' for column '{fk_col}'. Removing these rows.")
        # Remove rows with invalid FKs
        child_df = child_df[valid_fks].reset_index(drop=True)
    else:
        print(f"All foreign keys in '{child_table_name}' for column '{fk_col}' are valid.")

    return child_df

class CitationDataset(Dataset):
    name = "rel-citation"
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )
    # Train for the most recent 1 year out of 2 years of the original
    # time period
    # train_start_timestamp = pd.Timestamp("1937-01-01")  # TODO: replace after testing pd.Timestamp("1937-01-01") --pd.Timestamp("2020-01-01")
    val_timestamp = pd.Timestamp("2008-01-01")  # TODO: why the val is only one "dimedelta" and not more
    test_timestamp = pd.Timestamp("2014-01-02")
    # max_eval_time_frames = 1
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


        abstracts_expanded_df = pd.read_csv(abstracts_expanded, low_memory=False)
        authors_expanded_df = pd.read_csv(authors_expanded, low_memory=False)
        citations_expanded_df = pd.read_csv(citations_expanded, low_memory=False)
        paperids_expanded_df = pd.read_csv(paperids_expanded, low_memory=False)
        papers_expanded_df = pd.read_csv(papers_expanded, low_memory=False)
        # papers_expanded_df = papers_expanded_df[~pd.isna(papers_expanded_df.externalid_dblp)]  # Look only on papers that also appear in dblp

        default_date = self.train_start_timestamp

        # pre-processing the dataset
        dblp_publication_df["year"] = pd.to_datetime(dblp_publication_df["year"], format='%Y-%m-%d').fillna(default_date)
        # ssd_papers_df["publicationdate"] = pd.to_datetime(ssd_papers_df["publicationdate"], format='%Y-%m-%d').fillna(default_date)
        papers_expanded_df["year"] = pd.to_datetime(papers_expanded_df["year"], format='%Y-%m-%d').fillna(default_date)

        # Create the mapping dictionary
        pubid_to_pubkey = pd.Series(
            dblp_publication_df["pubkey"].values,
            index=dblp_publication_df["pubid"].astype(str)
        ).to_dict()

        # Define which columns in each DataFrame contain 'pubid'
        pubid_columns_mapping = {
            "dblp_publication_df": [],  # Already mapped, no replacement needed
            "dblp_article_df": ["pubid"],
            "dblp_book_df": ["pubid"],
            "dblp_incollection_df": ["pubid"],
            "dblp_inproceedings_df": ["pubid"],
            "dblp_authored_df": ["pubid"],  # Assuming 'pubid' is a column here
            # Add other DataFrames and their pubid columns as necessary
        }

        # List of DataFrames and their variable names
        dataframes = {
            "dblp_article_df": dblp_article_df,
            "dblp_book_df": dblp_book_df,
            "dblp_incollection_df": dblp_incollection_df,
            "dblp_inproceedings_df": dblp_inproceedings_df,
            "dblp_authored_df": dblp_authored_df,
            "abstracts_expanded_df": abstracts_expanded_df,
            "authors_expanded_df": authors_expanded_df,
            "citations_expanded_df": citations_expanded_df,
            # "paperids_expanded_df":paperids_expanded_df,
            "papers_expanded_df": papers_expanded_df
        }

        # Perform the replacement
        for df_name, df in dataframes.items():
            cols = pubid_columns_mapping.get(df_name, [])
            if cols:
                dataframes[df_name] = replace_pubid_with_pubkey(df, cols, pubid_to_pubkey)


        # added another key_ column
        # dblp_publication_df["key_pubkey"] = dblp_publication_df["pubkey"].astype(str)
        # dblp_publication_df["key_pubid"] = dblp_publication_df["pubid"]  # pd.to_numeric(dblp_publication_df["pubid"], errors='coerce')
        # dblp_article_df["key_pubid"] = dblp_article_df["pubid"]
        # dblp_book_df["key_pubid"] = dblp_book_df["pubid"]
        # dblp_incollection_df["key_pubid"] = dblp_incollection_df["pubid"]
        # dblp_inproceedings_df["key_pubid"] = dblp_inproceedings_df["pubid"]
        # dblp_author_df["key_authorid"] = dblp_author_df["authorid"]

        # abstracts_expanded_df["key_corpusid"] = abstracts_expanded_df["corpusid"]
        # authors_expanded_df["key_authorid"] = authors_expanded_df["authorid"]
        # citations_expanded_df["key_citationid"] = citations_expanded_df["citationid"]  # citationid is the key
        # citations_expanded_df["key_citingcorpusid"] = citations_expanded_df["citingcorpusid"]
        # citations_expanded_df["key_citedcorpusid"] = citations_expanded_df["citedcorpusid"]  # there are nulls
        # paperids_expanded_df["key_sha"] = paperids_expanded_df["sha"]
        papers_expanded_df["externalid_dblp"] = papers_expanded_df["externalid_dblp"].astype(str)
        papers_expanded_df = papers_expanded_df.rename(columns={"first_author_id": "authorId"})  # change the name of the column for convenience
        # papers_expanded_df["key_corpusid"] = papers_expanded_df["corpusid"]

        # ---- Referential Integrity Checks Start Here ----

        # 1. Ensure Parent Tables' Primary Keys are Unique
        parent_tables = {
            "dblp_publication": dblp_publication_df,
            "dblp_author": dblp_author_df,
            "authors_expanded": authors_expanded_df,
            "papers_expanded": papers_expanded_df,
            # Add other parent tables as necessary
        }

        # Mapping parent tables to their PK columns
        parent_pk = {
            "dblp_publication": "pubkey",
            "dblp_author": "authorid",
            "authors_expanded": "authorid",
            "papers_expanded": "corpusid",
            # Add other parent tables and their PK columns as necessary
        }

        for table_name, df in parent_tables.items():
            pk = parent_pk.get(table_name)
            if pk:
                if df[pk].duplicated().any():
                    duplicates = df[df[pk].duplicated(keep=False)]
                    raise ValueError(f"Primary key '{pk}' in table '{table_name}' has duplicates:\n{duplicates}")
                else:
                    print(f"Primary key '{pk}' in table '{table_name}' is unique.")

        # 2. Define Foreign Key Relationships
        # Create a dictionary mapping child table to its foreign keys and parent table
        foreign_keys = {
            "dblp_article_df": {"pubid": "dblp_publication"},
            "dblp_book_df": {"pubid": "dblp_publication"},
            "dblp_incollection_df": {"pubid": "dblp_publication"},
            "dblp_inproceedings_df": {"pubid": "dblp_publication"},
            "dblp_authored_df": {"pubid": "dblp_publication", "authorid": "dblp_author"},
            "abstracts_expanded_df": {"corpusid": "papers_expanded"},
            "citations_expanded_df": {"citingcorpusid": "papers_expanded", "citedcorpusid": "papers_expanded"},
            "papers_expanded_df": {"authorId": "authors_expanded", "externalid_dblp": "dblp_publication"},
            # Add other child tables and their foreign keys as necessary
        }

        # 3. Perform Referential Integrity Checks and Remove Invalid Rows
        for child_df_name, fk_dict in foreign_keys.items():
            child_df = dataframes.get(child_df_name)
            if child_df is None:
                print(f"Child DataFrame '{child_df_name}' not found. Skipping.")
                continue

            for fk_col, parent_table in fk_dict.items():
                parent_pk_col = parent_pk.get(parent_table)
                if not parent_pk_col:
                    raise ValueError(f"Parent table '{parent_table}' does not have a defined primary key.")

                parent_df = parent_tables[parent_table]

                # Enforce referential integrity by removing invalid rows
                child_df = enforce_referential_integrity(
                    child_df=child_df,
                    fk_col=fk_col,
                    parent_df=parent_df,
                    parent_pk_col=parent_pk_col,
                    child_table_name=child_df_name,
                    parent_table_name=parent_table
                )

                # Update the DataFrame in the dictionary
                dataframes[child_df_name] = child_df

        # Update DataFrame variables after integrity checks
        for df_name in dataframes:
            globals()[df_name] = dataframes[df_name]

        # ---- Referential Integrity Checks End Here ----

        return Database(
            table_dict={
                "dblp_publication": Table(
                    df=dblp_publication_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="pubkey",  # this is the key with semantic scholar, but the real pkey is pubid
                    time_col="year",
                ),
                "dblp_article": Table(
                    df=dblp_article_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col=None,  # "key_pubid",
                    time_col=None,
                ),
                "dblp_book": Table(
                    df=dblp_book_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col=None,  # "key_pubid",
                    time_col=None,
                ),
                "dblp_incollection": Table(
                    df=dblp_incollection_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col=None,  # "pubid",
                    time_col=None,
                ),
                "dblp_inproceedings": Table(
                    df=dblp_inproceedings_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication"},
                    pkey_col=None,  # "key_pubid",
                    time_col=None,
                ),
                "dblp_author": Table(
                    df=dblp_author_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="authorid",
                    time_col=None,
                ),
                "dblp_authored": Table(
                    df=dblp_authored_df,
                    fkey_col_to_pkey_table={"pubid": "dblp_publication", "authorid": "dblp_author"},
                    pkey_col=None,
                    time_col=None,
                ),

                # TODO: fill the end of the dataset correctly with the foreign keys and all.
                # Expanded Tables
                "abstracts_expanded": Table(
                    df=abstracts_expanded_df,
                    fkey_col_to_pkey_table={"corpusid": "papers_expanded"},
                    pkey_col=None,  # "corpusid",
                    time_col=None,
                ),
                "authors_expanded": Table(
                    df=authors_expanded_df,
                    fkey_col_to_pkey_table={},  # No foreign keys
                    pkey_col="authorid",
                    time_col=None,
                ),
                "citations_expanded": Table(
                    df=citations_expanded_df,
                    fkey_col_to_pkey_table={
                        "citingcorpusid": "papers_expanded",
                        "citedcorpusid": "papers_expanded",
                    },  # FKs to papers_expanded
                    pkey_col=None,  # "citationid",
                    time_col=None,
                ),
                # "paperids_expanded": Table(  # We dont use the paperids table for now
                #     df=paperids_expanded_df,
                #     fkey_col_to_pkey_table={"key_corpusid": "papers_expanded"},
                #     pkey_col="key_sha",
                #     time_col=None,
                # ),
                "papers_expanded": Table(
                    df=papers_expanded_df,
                    fkey_col_to_pkey_table={"authorId": "authors_expanded", "externalid_dblp": "dblp_publication"},  # No foreign keys
                    pkey_col="corpusid",  # externalid_dblp
                    time_col="year",
                ),
                # "ssd_papers": Table(
                #     df=ssd_papers_df,
                #     fkey_col_to_pkey_table={},
                #     pkey_col="key_dblp_external_id",
                #     time_col="publicationdate",
                # ),
            }
        )
