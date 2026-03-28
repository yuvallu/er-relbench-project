"""
Create a small POC subset of the DBLP-Semantic Scholar ER dataset.
Samples papers from papers_expanded that have valid dblp matches,
then ensures dblp_publication contains ALL referenced pubkeys.
"""
import pandas as pd
import os
import numpy as np

BASE_PATH = "/data/home/lubarsky/phd/s2ag-corpus/base_dir/csvs_dataset"
OUT_PATH = "/data/home/lubarsky/phd/er_project/poc_data"
os.makedirs(OUT_PATH, exist_ok=True)

SEED = 42
N_PAPERS = 5000

print("Loading papers_expanded...")
papers_expanded = pd.read_csv(os.path.join(BASE_PATH, "papers_expanded.csv"), low_memory=False)
print(f"  Total: {len(papers_expanded)} rows")

print("Loading dblp_publication...")
dblp_pub_full = pd.read_csv(os.path.join(BASE_PATH, "dblp_publication.csv"), low_memory=False)
print(f"  Total: {len(dblp_pub_full)} rows")

# Get the set of valid dblp pubkeys
valid_pubkeys = set(dblp_pub_full["pubkey"].dropna())

# Filter papers that have a valid DBLP external ID that exists in dblp_publication
papers_with_dblp = papers_expanded[
    papers_expanded["externalid_dblp"].notna() &
    (papers_expanded["externalid_dblp"] != "") &
    (papers_expanded["externalid_dblp"] != "nan") &
    papers_expanded["externalid_dblp"].isin(valid_pubkeys)
].copy()
print(f"  Papers with valid DBLP match: {len(papers_with_dblp)}")

# Sample papers with DBLP IDs
np.random.seed(SEED)
sampled_with_dblp = papers_with_dblp.sample(n=min(N_PAPERS, len(papers_with_dblp)), random_state=SEED)

# Add papers without DBLP IDs as negatives
papers_without_dblp = papers_expanded[
    papers_expanded["externalid_dblp"].isna() |
    (papers_expanded["externalid_dblp"] == "") |
    (papers_expanded["externalid_dblp"] == "nan")
]
sampled_without_dblp = papers_without_dblp.sample(n=min(N_PAPERS // 2, len(papers_without_dblp)), random_state=SEED)

sampled_papers = pd.concat([sampled_with_dblp, sampled_without_dblp], ignore_index=True)
print(f"  Sampled papers_expanded: {len(sampled_papers)} ({len(sampled_with_dblp)} with DBLP, {len(sampled_without_dblp)} without)")

corpus_ids = set(sampled_papers["corpusid"].dropna().astype(int))
dblp_keys_needed = set(sampled_papers["externalid_dblp"].dropna())
dblp_keys_needed.discard("")
dblp_keys_needed.discard("nan")

# dblp_publication MUST contain all referenced pubkeys from papers_expanded
matched_pubs = dblp_pub_full[dblp_pub_full["pubkey"].isin(dblp_keys_needed)]
print(f"  DBLP pubs that match sampled papers: {len(matched_pubs)}")

# Add distractors
unmatched_pubs = dblp_pub_full[~dblp_pub_full["pubkey"].isin(dblp_keys_needed)].sample(
    n=min(N_PAPERS, len(dblp_pub_full) - len(matched_pubs)), random_state=SEED
)
sampled_dblp_pub = pd.concat([matched_pubs, unmatched_pubs], ignore_index=True)
print(f"  Sampled dblp_publication: {len(sampled_dblp_pub)} ({len(matched_pubs)} matched, {len(unmatched_pubs)} distractors)")

pubkeys = set(sampled_dblp_pub["pubkey"].dropna())
pubids = set(sampled_dblp_pub["pubid"].dropna().astype(int))

# Load and filter authors_expanded (must contain all author IDs referenced by papers)
print("Loading authors_expanded...")
authors_exp = pd.read_csv(os.path.join(BASE_PATH, "authors_expanded.csv"), low_memory=False)
author_ids_needed = set(sampled_papers["first_author_id"].dropna().astype(int))
sampled_authors_exp = authors_exp[authors_exp["authorid"].isin(author_ids_needed)]
print(f"  Sampled authors_expanded: {len(sampled_authors_exp)} from {len(authors_exp)}")

# Remove papers whose author is not in authors_expanded
valid_author_ids = set(sampled_authors_exp["authorid"])
# For papers without author, keep them (authorId will be NaN which is ok)
sampled_papers_clean = sampled_papers[
    sampled_papers["first_author_id"].isna() |
    sampled_papers["first_author_id"].astype(float).astype('Int64').isin(valid_author_ids)
]
print(f"  Papers after author filter: {len(sampled_papers_clean)} (from {len(sampled_papers)})")
sampled_papers = sampled_papers_clean
corpus_ids = set(sampled_papers["corpusid"].dropna().astype(int))

# dblp_authored
print("Loading dblp_authored...")
dblp_authored = pd.read_csv(os.path.join(BASE_PATH, "dblp_authored.csv"))
sampled_authored = dblp_authored[dblp_authored["pubid"].isin(pubids)]
print(f"  Sampled dblp_authored: {len(sampled_authored)}")

# dblp_author (must contain all authors referenced by dblp_authored)
author_ids_dblp = set(sampled_authored["authorid"].dropna().astype(int))
print("Loading dblp_author...")
dblp_author = pd.read_csv(os.path.join(BASE_PATH, "dblp_author.csv"))
sampled_dblp_author = dblp_author[dblp_author["authorid"].isin(author_ids_dblp)]
print(f"  Sampled dblp_author: {len(sampled_dblp_author)}")

# citations_expanded
print("Loading citations_expanded...")
citations = pd.read_csv(os.path.join(BASE_PATH, "citations_expanded.csv"), low_memory=False)
sampled_citations = citations[
    citations["citingcorpusid"].isin(corpus_ids) &
    citations["citedcorpusid"].isin(corpus_ids)
]
print(f"  Sampled citations_expanded: {len(sampled_citations)}")

# abstracts_expanded
print("Loading abstracts_expanded...")
abstracts = pd.read_csv(os.path.join(BASE_PATH, "abstracts_expanded.csv"), low_memory=False)
sampled_abstracts = abstracts[abstracts["corpusid"].isin(corpus_ids)]
print(f"  Sampled abstracts_expanded: {len(sampled_abstracts)}")

# dblp sub-tables
for tbl_name in ["dblp_article", "dblp_book", "dblp_incollection", "dblp_inproceedings"]:
    print(f"Loading {tbl_name}...")
    df = pd.read_csv(os.path.join(BASE_PATH, f"{tbl_name}.csv"), low_memory=False)
    sampled = df[df["pubid"].isin(pubids)]
    sampled.to_csv(os.path.join(OUT_PATH, f"{tbl_name}.csv"), index=False)
    print(f"  Sampled {tbl_name}: {len(sampled)}")

# Save all
sampled_papers.to_csv(os.path.join(OUT_PATH, "papers_expanded.csv"), index=False)
sampled_dblp_pub.to_csv(os.path.join(OUT_PATH, "dblp_publication.csv"), index=False)
sampled_authored.to_csv(os.path.join(OUT_PATH, "dblp_authored.csv"), index=False)
sampled_dblp_author.to_csv(os.path.join(OUT_PATH, "dblp_author.csv"), index=False)
sampled_citations.to_csv(os.path.join(OUT_PATH, "citations_expanded.csv"), index=False)
sampled_authors_exp.to_csv(os.path.join(OUT_PATH, "authors_expanded.csv"), index=False)
sampled_abstracts.to_csv(os.path.join(OUT_PATH, "abstracts_expanded.csv"), index=False)

print("\n=== POC Dataset Summary ===")
for f in sorted(os.listdir(OUT_PATH)):
    if f.endswith(".csv"):
        n = sum(1 for _ in open(os.path.join(OUT_PATH, f))) - 1
        print(f"  {f}: {n} rows")
print("Done!")
