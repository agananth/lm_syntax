import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

import dataclasses
from collections.abc import Sequence, Mapping
from sklearn.metrics import f1_score, accuracy_score
import os
import cache_hidden_states
import dataset
import utils
from torch.utils.data import DataLoader
from typing import Any
from einops import rearrange
import train
import numpy as np
import copy


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[str, str]]


def _run_eval_inference(
    probe_train_states, data_loader, split_name, root_hidden_states
):
    flattened_labels = []
    flattened_preds = [[] for _ in probe_train_states]
    losses = [[] for _ in probe_train_states]
    for batch in data_loader:
        hidden_states, labels = batch
        hidden_states = rearrange(hidden_states, "1 w l h -> l w h")
        hidden_states = torch.cat((root_hidden_states, hidden_states), dim=1).cuda()
        labels = torch.tensor([0] + labels)
        flattened_labels.extend(labels.tolist())
        labels = labels.cuda()
        for j, probe_train_state in enumerate(probe_train_states):
            layer = probe_train_state.layer
            probe = probe_train_state.probe
            probe.eval()
            with torch.inference_mode():
                logits = probe(hidden_states[layer])
            distances = -torch.cdist(logits, logits)
            # Exclude root from masking
            mask = F.pad(torch.eye(distances.shape[0] - 1), (1, 0, 1, 0)).bool()
            distances[mask] = float("-inf")
            preds = distances.softmax(dim=-1).argmax(dim=-1)
            flattened_preds[j].extend(preds.tolist())
            losses[j].append(F.cross_entropy(distances, labels).item())

    metrics = []
    for i, probe_train_state in enumerate(probe_train_states):
        loss = torch.tensor(losses[i]).mean()
        flattened_class_preds = flattened_preds[i]
        layer = probe_train_state.layer
        layer_metrics = {
            f"layer_{layer}/{split_name}/loss": loss.item(),
            f"layer_{layer}/{split_name}/f1_micro": f1_score(
                flattened_labels,
                flattened_class_preds,
                average="micro",
                zero_division=np.nan,
            ),
            f"layer_{layer}/{split_name}/acc": accuracy_score(
                flattened_labels, flattened_class_preds
            ),
        }
        metrics.append(layer_metrics)
    return metrics


def train_probes(
    probe_train_states: Sequence[train.ProbeTrainState],
    config: dict[str, Any],
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    root_hidden_states,
):
    eval_interval = config["eval_interval"]
    grad_accum_steps = config["batch_size"]
    for epoch in range(config["max_epochs"]):
        for batch in train_data_loader:
            probes_to_eval = []
            for probe_train_state in probe_train_states:
                if probe_train_state.training_complete:
                    continue
                if probe_train_state.step % eval_interval == 0:
                    probes_to_eval.append(probe_train_state)
            if probes_to_eval:
                metrics = _run_eval_inference(
                    probe_train_states=probes_to_eval,
                    data_loader=val_data_loader,
                    split_name="val",
                    root_hidden_states=root_hidden_states,
                )
                for probe_state, metric in zip(probes_to_eval, metrics):
                    wandb.log(metric, step=probe_state.step)
                    early_stopping_metric = metric[
                        f"layer_{probe_state.layer}/val/{probe_state.early_stopping_metric.name}"
                    ]
                    is_lower_better = probe_state.early_stopping_metric.is_lower_better
                    if (
                        is_lower_better
                        and early_stopping_metric < probe_state.best_metric_value
                    ) or (
                        not is_lower_better
                        and early_stopping_metric > probe_state.best_metric_value
                    ):
                        probe_state.num_without_metric_improvement = 0
                        probe_state.best_probe = copy.deepcopy(probe_state.probe)
                        probe_state.best_metric_value = early_stopping_metric
                    else:
                        probe_state.num_without_metric_improvement += 1
                        if (
                            probe_state.num_without_metric_improvement
                            == config["patience"]
                        ):
                            print(
                                f"Layer {probe_state.layer} early stopping at step {probe_state.step}"
                            )
                            probe_state.training_complete = True
            remaining_probe_train_states = [
                probe_train_state
                for probe_train_state in probe_train_states
                if not probe_train_state.training_complete
            ]
            if not remaining_probe_train_states:
                print("All probes have completed training")
                break
            hidden_states, labels = batch
            # w = num_words
            hidden_states = rearrange(hidden_states, "1 w l h -> l w h")
            # add the root to the hidden states
            hidden_states = torch.cat((root_hidden_states, hidden_states), dim=1).cuda()
            labels = torch.tensor([0] + labels).cuda()
            for probe_train_state in remaining_probe_train_states:
                layer = probe_train_state.layer
                probe = probe_train_state.probe
                probe.train()

                logits = probe(hidden_states[layer])
                distances = -torch.cdist(logits, logits)
                # Exclude root from masking
                mask = F.pad(torch.eye(distances.shape[0] - 1), (1, 0, 1, 0)).bool()
                distances[mask] = float("-inf")
                loss = F.cross_entropy(distances, labels)
                probe_train_state.train_losses.append(loss.item())
                loss = loss / grad_accum_steps
                loss.backward()
                if (probe_train_state.step + 1) % grad_accum_steps == 0:
                    optimizer = probe_train_state.optimizer
                    optimizer.step()
                    optimizer.zero_grad()

                    wandb.log(
                        {
                            f"layer_{probe_train_state.layer}/train/loss": torch.tensor(
                                probe_train_state.train_losses
                            ).mean()
                        },
                        step=probe_train_state.step,
                    )
                    probe_train_state.train_losses.clear()
                probe_train_state.step += 1
        else:
            print(f"Completed epoch {epoch}")
            continue
        break

    model_name = config["model_name"]
    dir_name = os.path.join("head_word", model_name.replace("/", "_"))
    os.makedirs(dir_name, exist_ok=True)
    for probe_train_state in probe_train_states:
        f = os.path.join(dir_name, f"layer_{probe_train_state.layer}.pt")
        torch.save(
            probe_train_state.best_probe.state_dict(),
            f,
        )
        artifact = wandb.Artifact(
            f"{model_name.replace("/", "_")}_probe_layer_{probe_train_state.layer}", type="model"
        )
        artifact.add_file(f)
        wandb.log_artifact(artifact)
        wandb.config[f'layer_{probe_train_state.layer}/best_acc'] = probe_train_state.best_metric_value
        wandb.config[f'layer_{probe_train_state.layer}/step'] = probe_train_state.step - config["patience"]
    wandb.finish()


def main(parser):
    args = parser.parse_args()
    model_name = args.model

    tokenizer = utils.get_tokenizer(model_name)
    model = utils.get_model(model_name)
    num_layers = utils.get_num_layers(model.config)
    model.eval()
    hidden_size = model.config.hidden_size

    root_hidden_state = cache_hidden_states.get_word_hidden_states(
        ["ROOT"], tokenizer, model
    )[0]
    root_hidden_state = torch.cat(root_hidden_state, dim=0).unsqueeze(1)
    assert root_hidden_state.shape == (num_layers, 1, hidden_size)
    probe_hidden_size = 256

    config = dict(
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        model_name=model_name,
        hidden_size=model.config.hidden_size,
        num_layers=num_layers,
        eval_interval=args.batch_size * 2,
        log_interval=args.batch_size,
        probe_hidden_size=probe_hidden_size,
    )

    train_data_loader = DataLoader(
        dataset.HeadWordDataset(
            split_name="train",
            model_name=model_name,
            num_layers=num_layers,
            hidden_size=model.config.hidden_size,
        ),
        batch_size=1,
        shuffle=True,
    )

    val_data_loader = DataLoader(
        dataset.HeadWordDataset(
            split_name="dev",
            model_name=model_name,
            num_layers=num_layers,
            hidden_size=model.config.hidden_size,
        ),
        batch_size=1,
        shuffle=False,
    )

    wandb.init(project="Head Word Final", name=model_name, config=config)

    probe_train_states = []
    for layer in range(num_layers):
        probe = nn.Linear(
            in_features=hidden_size, out_features=probe_hidden_size, bias=False
        )
        probe.cuda()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        probe_train_states.append(
            train.ProbeTrainState(
                layer=layer,
                probe=probe,
                optimizer=optimizer,
                early_stopping_metric=train.EarlyStoppingMetric(
                    name="acc", is_lower_better=False
                ),
                best_metric_value=float("-inf"),
            )
        )

    train_probes(
        probe_train_states=probe_train_states,
        config=config,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        root_hidden_states=root_hidden_state,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="HF base model name"
    )
    parser.add_argument("-batch_size", type=int, default=1024)
    parser.add_argument("-lr", type=float, default=1e-2)
    parser.add_argument("-max_epochs", type=int, default=10)
    parser.add_argument("-patience", type=int, default=20)

    main(parser)
