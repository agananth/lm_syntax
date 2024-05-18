import torch
import run_registry
import wandb
import utils
from torch.utils.data import DataLoader
import dataset
from transformers import AutoConfig
from einops import rearrange
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import collections
import dataclasses
from collections.abc import Sequence, Mapping


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[str, str]]


def main():
    for model_name in run_registry.RUNS.keys():
        config = AutoConfig.from_pretrained(model_name)
        data_loader = DataLoader(
            dataset.HeadWordDatasetWithRelns(
                split_name="test",
                model_name=model_name,
                num_layers=utils.get_num_layers(config),
                hidden_size=config.hidden_size,
            ),
            batch_size=1,
            shuffle=False,
        )

        num_layers = utils.get_num_layers(config)

        flattened_labels = []
        flattened_cosine_preds = [[] for _ in range(num_layers)]
        flattened_euc_preds = [[] for _ in range(num_layers)]
        flattened_relns = []
        for batch in data_loader:
            hidden_states, labels_and_relns = batch
            labels, relns = zip(*labels_and_relns)
            relns = [rel[0] for rel in relns]
            flattened_relns.extend(relns)
            hidden_states = rearrange(hidden_states, "1 w l h -> l w h")
            labels = torch.tensor(labels)
            flattened_labels.extend(labels.tolist())
            for layer in range(num_layers):
                layer_hidden_states = hidden_states[layer]
                # cosine
                cosines = cosine_similarity(layer_hidden_states, layer_hidden_states)
                mask = np.eye(cosines.shape[0], dtype=bool)
                cosines[mask] = float("-inf")
                cosine_preds = cosines.argmax(axis=-1) + 1
                flattened_cosine_preds[layer].extend(cosine_preds.tolist())

                distances = -torch.cdist(layer_hidden_states, layer_hidden_states)
                mask = torch.eye(distances.shape[0]).bool()
                distances[mask] = float("-inf")
                euc_preds = distances.softmax(dim=-1).argmax(dim=-1) + 1
                flattened_euc_preds[layer].extend(euc_preds.tolist())

        hidden_size = config.hidden_size
        wandb.init(
            project="Head Word Unsupervised",
            config=dict(num_layers=num_layers, hidden_size=hidden_size),
            name=model_name,
        )
        cosine_accuracies = []
        for layer in range(num_layers):
            labels_to_count = []
            cos_predictions_to_count = []
            euc_predictions_to_count = []
            for label, cos_pred, euc_pred, reln in zip(
                flattened_labels,
                flattened_cosine_preds[layer],
                flattened_euc_preds[layer],
                flattened_relns,
            ):
                if reln == "root" or reln == "punct":
                    continue
                labels_to_count.append(label)
                cos_predictions_to_count.append(cos_pred)
                euc_predictions_to_count.append(euc_pred)
            cosine_accuracy = accuracy_score(labels_to_count, cos_predictions_to_count)
            euc_accuracy = accuracy_score(labels_to_count, euc_predictions_to_count)
            wandb.log(
                {
                    f"layer_{layer}_cosine_accuracy": cosine_accuracy,
                    f"layer_{layer}_euc_accuracy": euc_accuracy,
                },
                step=1,
            )
            cosine_accuracies.append(cosine_accuracy)

        best_cosine_layer = np.argmax(cosine_accuracies)
        best_cosine_preds = flattened_cosine_preds[best_cosine_layer]
        total_counter = collections.Counter()
        correct_counter = collections.Counter()
        for label, pred, reln in zip(
            flattened_labels, best_cosine_preds, flattened_relns
        ):
            if reln == "root" or reln == "punct":
                continue
            total_counter[reln] += 1
            if label == pred:
                correct_counter[reln] += 1

        for reln in total_counter.keys():
            wandb.summary[f"{reln}_total"] = total_counter[reln]
            wandb.summary[f"{reln}_correct"] = correct_counter[reln]
        wandb.finish()


if __name__ == "__main__":
    main()
