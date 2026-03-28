"""
Create a larger POC subset with temporal splits that actually produce train/val/test data.

Strategy:
- Sample 15K papers with valid DBLP matches spread across years
- Add 5K distractor papers without DBLP matches
- Include all matching DBLP publications + distractors
- Temporal splits: val=2019, test=2022 (median matched paper is 2019)
"""
import pandas as pd
import os
import numpy as np

BASE_PATH = "/data/home/lubarsky/phd/s2ag-corpus/base_dir/csvs_dataset"
OUT_PATH = "/data/home/lubarsky/phd/er_project/poc_data_v2"
os.makedirs(OUT_PATH, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# === Load papers_expanded ===
print("Loading papers_expanded...")
papers_expanded = pd.read_csv(os.path.join(BASE_PATH, "papers_expanded.csv"), low_memory=False)
papers_expanded['year_dt'] = pd.to_datetime(papers_expanded['year'], errors='coerce')
papers_expanded['yr'] = papers_expanded['year_dt'].dt.year
print(f"  Total: {len(papers_expanded)} rows")

# === Load dblp_publication ===
print("Loading dblp_publication...")
dblp_pub_full = pd.read_csv(os.path.join(BASE_PATH, "dblp_publication.csv"), low_memory=False)
valid_pubkeys = set(dblp_pub_full["pubkey"].dropna())
print(f"  Total: {len(dblp_pub_full)} rows")

# === Filter to papers with valid DBLP match ===
has_dblp = (
    papers_expanded["externalid_dblp"].notna() &
    (papers_expanded["externalid_dblp"] != "") &
    (papers_expanded["externalid_dblp"].astype(str) != "nan") &
    papers_expanded["externalid_dblp"].isin(valid_pubkeys)
)
papers_with_dblp = papers_expanded[has_dblp].copy()
print(f"  Papers with valid DBLP match: {len(papers_with_dblp)}")

# === Stratified sampling by year to ensure good temporal coverage ===
# We want papers from pre-2019 (train), 2019-2021 (val), and 2022+ (test)
train_papers = papers_with_dblp[papers_with_dblp['yr'] < 2019]
val_papers = papers_with_dblp[(papers_with_dblp['yr'] >= 2019) & (papers_with_dblp['yr'] < 2022)]
test_papers = papers_with_dblp[papers_with_dblp['yr'] >= 2022]

print(f"  Train pool (pre-2019): {len(train_papers)}")
print(f"  Val pool (2019-2021): {len(val_papers)}")
print(f"  Test pool (2022+): {len(test_papers)}")

# Sample from each era
n_train = 8000
n_val = 4000
n_test = 3000

sampled_train = train_papers.sample(n=min(n_train, len(train_papers)), random_state=SEED)
sampled_val = val_papers.sample(n=min(n_val, len(val_papers)), random_state=SEED)
sampled_test = test_papers.sample(n=min(n_test, len(test_papers)), random_state=SEED)

sampled_with_dblp = pd.concat([sampled_train, sampled_val, sampled_test], ignore_index=True)
print(f"  Sampled with DBLP: {len(sampled_with_dblp)} (train={len(sampled_train)}, val={len(sampled_val)}, test={len(sampled_test)})")

# Add distractors (no DBLP match)
papers_without_dblp = papers_expanded[~has_dblp]
sampled_without_dblp = papers_without_dblp.sample(n=min(5000, len(papers_without_dblp)), random_state=SEED)

sampled_papers = pd.concat([sampled_with_dblp, sampled_without_dblp], ignore_index=True)
# Drop helper columns
sampled_papers = sampled_papers.drop(columns=['year_dt', 'yr'], errors='ignore')
print(f"  Total sampled papers_expanded: {len(sampled_papers)}")

corpus_ids = set(sampled_papers["corpusid"].dropna().astype(int))
dblp_keys_needed = set(sampled_papers["externalid_dblp"].dropna())
dblp_keys_needed.discard("")
dblp_keys_needed.discard("nan")

# === dblp_publication: all matched + distractors ===
matched_pubs = dblp_pub_full[dblp_pub_full["pubkey"].isin(dblp_keys_needed)]
unmatched_pubs = dblp_pub_full[~dblp_pub_full["pubkey"].isin(dblp_keys_needed)].sample(
    n=min(10000, len(dblp_pub_full) - len(matched_pubs)), random_state=SEED
)
sampled_dblp_pub = pd.concat([matched_pubs, unmatched_pubs], ignore_index=True)
print(f"  Sampled dblp_publication: {len(sampled_dblp_pub)} ({len(matched_pubs)} matched, {len(unmatched_pubs)} distractors)")

pubkeys = set(sampled_dblp_pub["pubkey"].dropna())
pubids = set(sampled_dblp_pub["pubid"].dropna().astype(int))

# === authors_expanded ===
print("Loading authors_expanded...")
authors_exp = pd.read_csv(os.path.join(BASE_PATH, "authors_expanded.csv"), low_memory=False)
author_ids_needed = set(sampled_papers["first_author_id"].dropna().astype(int))
sampled_authors_exp = authors_exp[authors_exp["authorid"].isin(author_ids_needed)]
print(f"  Sampled authors_expanded: {len(sampled_authors_exp)}")

# === dblp_authored ===
print("Loading dblp_authored...")
dblp_authored = pd.read_csv(os.path.join(BASE_PATH, "dblp_authored.csv"))
sampled_authored = dblp_authored[dblp_authored["pubid"].isin(pubids)]
print(f"  Sampled dblp_authored: {len(sampled_authored)}")

# === dblp_author ===
author_ids_dblp = set(sampled_authored["authorid"].dropna().astype(int))
print("Loading dblp_author...")
dblp_author = pd.read_csv(os.path.join(BASE_PATH, "dblp_author.csv"))
sampled_dblp_author = dblp_author[dblp_author["authorid"].isin(author_ids_dblp)]
print(f"  Sampled dblp_author: {len(sampled_dblp_author)}")

# === citations_expanded (both endpoints must be in our sample) ===
print("Loading citations_expanded...")
citations = pd.read_csv(os.path.join(BASE_PATH, "citations_expanded.csv"), low_memory=False)
sampled_citations = citations[
    citations["citingcorpusid"].isin(corpus_ids) &
    citations["citedcorpusid"].isin(corpus_ids)
]
print(f"  Sampled citations_expanded: {len(sampled_citations)}")

# === abstracts_expanded ===
print("Loading abstracts_expanded...")
abstracts = pd.read_csv(os.path.join(BASE_PATH, "abstracts_expanded.csv"), low_memory=False)
sampled_abstracts = abstracts[abstracts["corpusid"].isin(corpus_ids)]
print(f"  Sampled abstracts_expanded: {len(sampled_abstracts)}")

# === dblp sub-tables ===
for tbl_name in ["dblp_article", "dblp_book", "dblp_incollection", "dblp_inproceedings"]:
    print(f"Loading {tbl_name}...")
    df = pd.read_csv(os.path.join(BASE_PATH, f"{tbl_name}.csv"), low_memory=False)
    sampled = df[df["pubid"].isin(pubids)]
    sampled.to_csv(os.path.join(OUT_PATH, f"{tbl_name}.csv"), index=False)
    print(f"  Sampled {tbl_name}: {len(sampled)}")

# === Save all ===
sampled_papers.to_csv(os.path.join(OUT_PATH, "papers_expanded.csv"), index=False)
sampled_dblp_pub.to_csv(os.path.join(OUT_PATH, "dblp_publication.csv"), index=False)
sampled_authored.to_csv(os.path.join(OUT_PATH, "dblp_authored.csv"), index=False)
sampled_dblp_author.to_csv(os.path.join(OUT_PATH, "dblp_author.csv"), index=False)
sampled_citations.to_csv(os.path.join(OUT_PATH, "citations_expanded.csv"), index=False)
sampled_authors_exp.to_csv(os.path.join(OUT_PATH, "authors_expanded.csv"), index=False)
sampled_abstracts.to_csv(os.path.join(OUT_PATH, "abstracts_expanded.csv"), index=False)

print("\n=== POC Dataset v2 Summary ===")
for f in sorted(os.listdir(OUT_PATH)):
    if f.endswith(".csv"):
        n = sum(1 for _ in open(os.path.join(OUT_PATH, f))) - 1
        print(f"  {f}: {n:,} rows")
print("Done!")
