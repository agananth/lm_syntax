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


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[str, str]]


api = wandb.Api()

for model_name, wandb_run_id in run_registry.RUNS.items():
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
        ),
        batch_size=1,
        shuffle=False,
    )
    probes = []
    for layer in range(num_layers):
        probe = nn.Linear(in_features=hidden_size, out_features=256, bias=False)
        artifact = api.artifact(
            f"ananthag/Head Word Final 2/{model_name.replace('/', '_')}_probe_layer_{layer}:v0"
        )
        path = artifact.download() + f"/layer_{layer}.pt"
        probe.load_state_dict(torch.load(path))
        probe.eval()
        probe.cuda()
        probes.append(probe)

    flattened_labels = []
    flattened_preds = [[] for _ in range(num_layers)]
    flattened_relns = []
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
    run = api.run(f"ananthag/Head Word Final 2/{wandb_run_id}")
    summary_metrics = run.summary_metrics
    for layer, preds in enumerate(flattened_preds):
        accuracy = accuracy_score(flattened_labels, preds)
        assert accuracy == summary_metrics[f"layer_{layer}/test/acc"], (
            accuracy,
            summary_metrics[f"layer_{layer}/test/acc"],
        )
        accuracies.append(accuracy)
    best_layer = np.argmax(accuracies)
    best_preds = flattened_preds[best_layer]
    total_counter = collections.Counter()
    correct_counter = collections.Counter()
    for label, pred, reln in zip(flattened_labels, best_preds, flattened_relns):
        if reln == "ROOT":
            continue
        total_counter[reln] += 1
        if label == pred:
            correct_counter[reln] += 1

    for reln in total_counter.keys():
        run.summary[f"{reln}_total"] = total_counter[reln]
        run.summary[f"{reln}_correct"] = correct_counter[reln]
    run.summary.update()
    print("Completed", model_name)
    del model
