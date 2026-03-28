# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""D3: A Massive Dataset of Scholarly Metadata for Analyzing Computer Science Research"""

import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import datasets
from datasets.tasks import TextClassification
from lxml import etree

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{wahle-etal-2022-d3,
    title = "D3: A Massive Dataset of Scholarly Metadata for Analyzing the State of Computer Science Research",
    author = "Wahle, Jan Philip  and
      Ruas, Terry  and
      Mohammad, Saif  and
      Gipp, Bela",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.283",
    pages = "2642--2651",
    abstract = "DBLP is the largest open-access repository of scientific articles on computer science and provides metadata associated with publications, authors, and venues. We retrieved more than 6 million publications from DBLP and extracted pertinent metadata (e.g., abstracts, author affiliations, citations) from the publication texts to create the DBLP Discovery Dataset (D3). D3 can be used to identify trends in research activity, productivity, focus, bias, accessibility, and impact of computer science research. We present an initial analysis focused on the volume of computer science research (e.g., number of papers, authors, research activity), trends in topics of interest, and citation patterns. Our findings show that computer science is a growing research field (15{\%} annually), with an active and collaborative researcher community. While papers in recent years present more bibliographical entries in comparison to previous decades, the average number of citations has been declining. Investigating papers{'} abstracts reveals that recent topic trends are clearly reflected in D3. Finally, we list further applications of D3 and pose supplemental research questions. The D3 dataset, our findings, and source code are publicly available for research purposes.",
}
"""

_DESCRIPTION = """This repository provides metadata to papers from DBLP."""

_HOMEPAGE = "https://github.com/jpwahle/lrec22-d3-dataset"

_LICENSE = (
    "DBLP Discovery Dataset (D3) is licensed under a Creative Commons"
    " Attribution-NonCommercial-ShareAlike 4.0 International License."
)

_URLS = [
    "https://zenodo.org/record/7071698/files/2022-11-30-authors.jsonl.gz?download=1"
    "https://zenodo.org/record/7071698/files/2022-11-30-papers.jsonl.gz?download=1",
]


class D3Config(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(
        self,
        features,
        data_url,
        data_dir,
        **kwargs,
    ):
        super(D3Config, self).__init__(
            version=datasets.Version("2.0.0", ""), **kwargs
        )
        self.features = features
        self.data_url = data_url
        self.data_dir = data_dir


class D3(datasets.GeneratorBasedBuilder):
    """D3 dataset."""

    BUILDER_CONFIGS = [
        D3Config(
            name="papers",
            features={
                "corpusid": datasets.Value("int64"),
                "title": datasets.Value("string"),
                "authors": datasets.Sequence(
                    {
                        "authorId": datasets.Value("int64"),
                        "name": datasets.Value("string"),
                    }
                ),
                "venue": datasets.Value("string"),
                "year": datasets.Value("int16"),
                "publicationdate": datasets.Value("string"),
                "abstract": datasets.Value("string"),
                "referencecount": datasets.Value("int64"),
                "citationcount": datasets.Value("int64"),
                "isopenaccess": datasets.Value("bool"),
                "influentialcitationcount": datasets.Value("int64"),
                "s2fieldsofstudy": datasets.Sequence(
                    {
                        "category": datasets.Value("string"),
                        "source": datasets.Value("string"),
                    }
                ),
                "publicationtypes": datasets.Sequence(
                    datasets.Value("string")
                ),
                "journal": datasets.Value("string"),
                "updated": datasets.Value("string"),
                "url": datasets.Value("string"),
                "externalids": {
                    "ACL": datasets.Value("string"),
                    "DBLP": datasets.Value("string"),
                    "ArXiv": datasets.Value("string"),
                    "MAG": datasets.Value("string"),
                    "CorpusId": datasets.Value("string"),
                    "PubMed": datasets.Value("string"),
                    "DOI": datasets.Value("string"),
                    "PubMedCentral": datasets.Value("string"),
                },
                "syntactic": datasets.Sequence(datasets.Value("string")),
                "semantic": datasets.Sequence(datasets.Value("string")),
                "union": datasets.Sequence(datasets.Value("string")),
                "enhanced": datasets.Sequence(datasets.Value("string"))
            },
            data_url="https://zenodo.org/record/7071698/files/2022-11-30-papers.jsonl.gz?download=1",
            data_dir="papers",
        ),
        D3Config(
            name="authors",
            features={
                "authorid": datasets.Value("int64"),
                "name": datasets.Value("string"),
                "homepage": datasets.Value("string"),
                "papercount": datasets.Value("int64"),
                "citationcount": datasets.Value("int64"),
                "hindex": datasets.Value("int64"),
                "aliases": datasets.Sequence(datasets.Value("string")),
                "affiliations": datasets.Sequence(datasets.Value("string")),
                "updated": datasets.Value("string"),
                "s2url": datasets.Value("string"),
                "externalids": {
                    "DBLP": datasets.Value("string"),
                    "ORCID": datasets.Value("string"),
                }
            },
            data_url="https://zenodo.org/record/7071698/files/2022-11-30-authors.jsonl.gz?download=1",
            data_dir="authors",
        ),
    ]

    def _info(self):
        features = datasets.Features(self.config.features)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_file = dl_manager.download_and_extract(self.config.data_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": dl_manager.iter_files(data_file),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepaths, split):
        """Yields examples."""
        for train_files in filepaths:
            with open(train_files, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    # For none open access papers, the abstract is not in the dataset
                    # Replace it with an empty string
                    if "abstract" not in data and self.config.name == "papers":
                        data["abstract"] = ""
                    yield id_, data
