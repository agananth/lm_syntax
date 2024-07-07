import numpy as np
from torch.utils.data import Dataset
import json


class BlimpDataset(Dataset):

    def __init__(self, file_path: str):
        self.good_sents = []
        self.bad_sents = []
        self.linguistics_term = None
        data = [json.loads(l) for l in open(file_path, encoding="utf-8").readlines()]
        for example in data:
            if not example["simple_LM_method"]:
                continue
            self.good_sents.append(example["sentence_good"])
            self.bad_sents.append(example["sentence_bad"])
            if self.linguistics_term is None:
                self.linguistics_term = example["linguistics_term"]
            else:
                assert self.linguistics_term == example["linguistics_term"]

    def __len__(self):
        return len(self.good_sents)

    def __getitem__(self, idx):
        return self.good_sents[idx], self.bad_sents[idx]


class BlimpDatasetWithWords(Dataset):

    def __init__(self, file_path: str):
        self.prefixes = []
        data = [json.loads(l) for l in open(file_path, encoding="utf-8").readlines()]
        for example in data:
            self.prefixes.append(example["one_prefix_word_good"])

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        return self.prefixes[idx]
