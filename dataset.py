import numpy as np
from torch.utils.data import Dataset
import utils
import os

_HIDDEN_STATE_CACHE_DIR = "/nlp/scr/ananthag/hidden_states"


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
