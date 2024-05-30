import numpy as np
from torch.utils.data import Dataset
import utils
import os
import stanza_cache
import dataclasses
from collections.abc import Sequence, Mapping
from stanza.models.constituency import parse_tree, tree_reader


_HIDDEN_STATE_CACHE_DIR = "/scr/biggest/ananthag/hidden_states"


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[str, str]]


class WordDataset(Dataset):

    def __init__(
        self,
        split_name: str,
        model_name: str,
        num_layers: int,
        hidden_size: int,
        label_name: str,
    ):
        self.ptb_csv = utils.read_csv(os.path.join("ptb", f"{split_name}.csv"))
        self.num_words = len(self.ptb_csv)
        self.hidden_state_cache = np.memmap(
            os.path.join(
                _HIDDEN_STATE_CACHE_DIR,
                model_name.replace("/", "_"),
                f"{split_name}.dat",
            ),
            "float32",
            mode="r",
            shape=(self.num_words, num_layers, hidden_size),
        )
        self.label_name = label_name

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.num_words

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data point.

        Returns:
            (Any, Any): A tuple containing the data sample and its label (or any relevant information)
        """
        return self.hidden_state_cache[idx], int(self.ptb_csv[idx][self.label_name])


class HeadWordDataset(Dataset):

    def __init__(
        self,
        split_name: str,
        model_name: str,
        num_layers: int,
        hidden_size: int,
        random_weights: bool,
    ):
        data = getattr(stanza_cache, f"cached_ptb_{split_name}")()
        self.start_indices = {}
        total_words = 0
        self.labels = []
        for i, dev_data in enumerate(data):
            self.start_indices[i] = total_words
            label_list = []
            for head in dev_data["heads"]:
                total_words += 1
                label_list.append(head)
            self.labels.append(label_list)
        self.hidden_state_cache = np.memmap(
            os.path.join(
                _HIDDEN_STATE_CACHE_DIR,
                "random" if random_weights else "",
                model_name.replace("/", "_"),
                # f"layers_0_{num_layers}",
                f"{split_name}.dat",
            ),
            "float32",
            mode="r",
            shape=(total_words, num_layers, hidden_size),
        )

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.start_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data point.

        Returns:
            (Any, Any): A tuple containing the data sample and its label (or any relevant information)
        """
        start_index = self.start_indices[idx]
        return (
            self.hidden_state_cache[start_index : start_index + len(self.labels[idx])],
            self.labels[idx],
        )


class HeadWordDatasetWithRelns(HeadWordDataset):

    def __init__(
        self,
        split_name: str,
        model_name: str,
        num_layers: int,
        hidden_size: int,
        random_weights: bool,
    ):
        data = getattr(stanza_cache, f"cached_ptb_{split_name}")()
        self.start_indices = {}
        total_words = 0
        self.labels = []
        for i, dev_data in enumerate(data):
            self.start_indices[i] = total_words
            label_list = []
            for head, reln in zip(dev_data["heads"], dev_data["relns"]):
                total_words += 1
                label_list.append((head, reln))
            self.labels.append(label_list)
        self.hidden_state_cache = np.memmap(
            os.path.join(
                _HIDDEN_STATE_CACHE_DIR,
                "random" if random_weights else "",
                model_name.replace("/", "_"),
                # f"layers_0_{num_layers}",
                f"{split_name}.dat",
            ),
            "float32",
            mode="r",
            shape=(total_words, num_layers, hidden_size),
        )
