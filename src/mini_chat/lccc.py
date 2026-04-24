# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""
LCCC: Large-scale Cleaned Chinese Conversation corpus (LCCC) is a large corpus of Chinese conversations.
A rigorous data cleaning pipeline is designed to ensure the quality of the corpus.
This pipeline involves a set of rules and several classifier-based filters.
Noises such as offensive or sensitive words, special symbols, emojis,
grammatically incorrect sentences, and incoherent conversations are filtered.
"""

import gzip
import json
import os

import datasets


# BibTeX citation
_CITATION = """\
@inproceedings{wang2020chinese,
title={A Large-Scale Chinese Short-Text Conversation Dataset},
author={Wang, Yida and Ke, Pei and Zheng, Yinhe and Huang, Kaili and Jiang, Yong and Zhu, Xiaoyan and Huang, Minlie},
booktitle={NLPCC},
year={2020},
url={https://arxiv.org/abs/2008.03946}
}
"""

# Description of the dataset here
_DESCRIPTION = """\
LCCC: Large-scale Cleaned Chinese Conversation corpus (LCCC) is a large corpus of Chinese conversations.
A rigorous data cleaning pipeline is designed to ensure the quality of the corpus.
This pipeline involves a set of rules and several classifier-based filters.
Noises such as offensive or sensitive words, special symbols, emojis,
grammatically incorrect sentences, and incoherent conversations are filtered.
"""

_HOMEPAGE = "https://github.com/thu-coai/CDial-GPT"
_LICENSE = "MIT"
# 远程 URL（需要网络）
# _URLS = {
#     "large": "https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_large.jsonl.gz",
#     "base": {
#         "train": "https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_base_train.jsonl.gz",
#         "valid": "https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_base_valid.jsonl.gz",
#         "test": "https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_base_test.jsonl.gz",
#     },
# }

# 本地路径（手动下载数据后使用）
_DATA_DIR = "/Users/sqb/Desktop/ai/jupter_ai/src/dataset"
_URLS = {
    "large": os.path.join(_DATA_DIR, "lccc_large.jsonl.gz"),
    "base": {
        "train": os.path.join(_DATA_DIR, "lccc_base_train.jsonl.gz"),
        "valid": os.path.join(_DATA_DIR, "lccc_base_valid.jsonl.gz"),
        "test": os.path.join(_DATA_DIR, "lccc_base_test.jsonl.gz"),
    },
}


class LCCC(datasets.GeneratorBasedBuilder):
    """Large-scale Cleaned Chinese Conversation corpus."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="large", version=VERSION, description="The large version of LCCC"),
        datasets.BuilderConfig(name="base", version=VERSION, description="The base version of LCCC"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "dialog": [datasets.Value("string")],
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        if self.config.name == "large":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": urls,
                        "split": "train",
                    },
                )
            ]
        if self.config.name == "base":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": urls["train"],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": urls["test"], "split": "test"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": urls["valid"],
                        "split": "dev",
                    },
                ),
            ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # 支持 .gz 压缩文件
        if filepath.endswith('.gz'):
            f = gzip.open(filepath, 'rt', encoding='utf-8')
        else:
            f = open(filepath, encoding='utf-8')

        with f:
            for key, row in enumerate(f):
                row = row.strip()
                if len(row) == 0:
                    continue
                yield key, {
                    "dialog": json.loads(row),
                }
