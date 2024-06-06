import wandb
import run_registry
import utils
import cache_hidden_states
import torch
from torch.utils.data import DataLoader
import dataset
import torch.nn as nn
import dataclasses
from collections.abc import Sequence, Mapping
import collections
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import argparse


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[str, str]]


api = wandb.Api()


def main(parser):
    args = parser.parse_args()
    use_random_model = args.random_model
    registry = (
        run_registry.RANDOM_WEIGHT_RUNS if use_random_model else run_registry.RUNS
    )
    for model_name, wandb_run_id in registry.items():
        tokenizer = utils.get_tokenizer(model_name)
        model = utils.get_model(model_name)
        model.eval()
        num_layers = utils.get_num_layers(model.config)
        hidden_size = model.config.hidden_size

        root_hidden_state = cache_hidden_states.get_word_hidden_states(
            ["ROOT"], tokenizer, model
        )[0]
        root_hidden_state = torch.cat(root_hidden_state, dim=0).unsqueeze(1)
        assert root_hidden_state.shape == (num_layers, 1, hidden_size)

        data_loader = DataLoader(
            dataset.HeadWordDatasetWithRelns(
                split_name="test",
                model_name=model_name,
                num_layers=num_layers,
                hidden_size=hidden_size,
                random_weights=use_random_model,
            ),
            batch_size=1,
            shuffle=False,
        )
        probes = []
        artifact_path_prefix = (
            "ananthag/Head Word Final w Base Weights"
            if use_random_model
            else "ananthag/Head Word Final 2"
        )
        for layer in range(num_layers):
            probe = nn.Linear(in_features=hidden_size, out_features=256, bias=False)
            artifact = api.artifact(
                f"{artifact_path_prefix}/{model_name.replace('/', '_')}_probe_layer_{layer}:v0"
            )
            path = artifact.download() + f"/layer_{layer}.pt"
            probe.load_state_dict(torch.load(path))
            probe.eval()
            probe.cuda()
            probes.append(probe)

        flattened_labels = []
        flattened_preds = [[] for _ in range(num_layers)]
        flattened_relns = []
        flattened_directions = []
        for batch in data_loader:
            hidden_states, labels_and_relns = batch
            labels, relns = zip(*labels_and_relns)
            relns = ["ROOT"] + [rel[0] for rel in relns]
            hidden_states = rearrange(hidden_states, "1 w l h -> l w h")
            hidden_states = torch.cat((root_hidden_state, hidden_states), dim=1).cuda()
            labels = torch.tensor([0] + list(labels))
            flattened_labels.extend(labels.tolist())
            labels = labels.cuda()
            flattened_relns.extend(relns)
            for position, l in enumerate(labels):
                if l == position:
                    flattened_directions.append("SELF")
                elif l < position:
                    flattened_directions.append("left")
                else:
                    flattened_directions.append("right")

            for layer, probe in enumerate(probes):
                with torch.inference_mode():
                    logits = probe(hidden_states[layer])
                distances = -torch.cdist(logits, logits)
                # Exclude root from masking
                mask = F.pad(torch.eye(distances.shape[0] - 1), (1, 0, 1, 0)).bool()
                distances[mask] = float("-inf")
                preds = distances.softmax(dim=-1).argmax(dim=-1)
                flattened_preds[layer].extend(preds.tolist())

        accuracies = []
        run = api.run(wandb_run_id)
        summary_metrics = run.summary_metrics
        for layer, preds in enumerate(flattened_preds):
            accuracy = accuracy_score(flattened_labels, preds)
            run.summary[f"layer_{layer}/test/acc"] = accuracy
            accuracies.append(accuracy)
        best_layer = np.argmax(accuracies)
        best_preds = flattened_preds[best_layer]
        total_counter = collections.Counter()
        correct_counter = collections.Counter()
        for label, pred, reln, direction in zip(
            flattened_labels, best_preds, flattened_relns, flattened_directions
        ):
            if reln == "ROOT":
                continue
            total_counter[reln] += 1
            total_counter[direction] += 1
            if reln == "nmod" and direction == "left":
                total_counter["nmod_left"] += 1
            elif reln == "nmod" and direction == "right":
                total_counter["nmod_right"] += 1
            if label == pred:
                correct_counter[reln] += 1
                correct_counter[direction] += 1
                if reln == "nmod" and direction == "left":
                    correct_counter["nmod_left"] += 1
                elif reln == "nmod" and direction == "right":
                    correct_counter["nmod_right"] += 1

        for reln in total_counter.keys():
            run.summary[f"{reln}_total"] = total_counter[reln]
            run.summary[f"{reln}_correct"] = correct_counter[reln]
        run.summary.update()
        print("Completed", model_name)
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_model", action="store_true")
    main(parser)
