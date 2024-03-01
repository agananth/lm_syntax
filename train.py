import collections
import copy
import dataclasses
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

import wandb


from transformers import set_seed

set_seed(42)


@dataclasses.dataclass
class EarlyStoppingMetric:
    name: str
    is_lower_better: bool


@dataclasses.dataclass
class ProbeTrainState:
    layer: int
    probe: nn.Module
    optimizer: torch.optim.Optimizer
    early_stopping_metric: EarlyStoppingMetric
    best_metric_value: float
    num_without_metric_improvement: int = 0
    best_probe: nn.Module | None = None
    step: int = 0
    training_complete: bool = False
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    train_losses: list[float] = dataclasses.field(default_factory=list)


def _calculate_per_cls_counts(flattened_preds, flattened_labels):
    per_class_totals = collections.Counter()
    per_class_corrects = collections.Counter()
    for pred, label in zip(flattened_preds, flattened_labels):
        per_class_totals[label] += 1
        if pred == label:
            per_class_corrects[label] += 1
    return per_class_totals, per_class_corrects


def _run_eval_inference(
    probe_train_states, data_loader, split_name, config, cls_to_tag
):
    flattened_labels = []
    flattened_preds = [[] for _ in probe_train_states]
    losses = [[] for _ in probe_train_states]
    for batch in data_loader:
        hidden_states, labels = batch
        hidden_states = rearrange(hidden_states, "b l h -> l b h").cuda()
        flattened_labels.extend(labels.tolist())
        labels = labels.cuda()
        for j, probe_train_state in enumerate(probe_train_states):
            layer = probe_train_state.layer
            probe = probe_train_state.probe
            probe.eval()
            with torch.inference_mode():
                logits = probe(hidden_states[layer])
            preds = logits.softmax(dim=-1).argmax(dim=-1)
            flattened_preds[j].extend(preds.tolist())
            losses[j].append(F.cross_entropy(logits, labels).item())

    metrics = []
    for i, probe_train_state in enumerate(probe_train_states):
        loss = torch.tensor(losses[i]).mean()
        flattened_class_preds = flattened_preds[i]
        per_class_totals, per_class_corrects = _calculate_per_cls_counts(
            flattened_class_preds, flattened_labels
        )
        layer = probe_train_state.layer
        per_cls_counts = {
            f"layer_{layer}/{tag}/count": (
                per_class_corrects[cls_idx],
                per_class_totals[cls_idx],
            )
            for cls_idx, tag in cls_to_tag.items()
        }
        labels = np.arange(config["num_classes"])
        layer_metrics = {
            f"layer_{layer}/{split_name}/loss": loss.item(),
            f"layer_{layer}/{split_name}/f1_macro": f1_score(
                flattened_labels,
                flattened_class_preds,
                labels=labels,
                average="macro",
                zero_division=np.nan,
            ),
            f"layer_{layer}/{split_name}/f1_micro": f1_score(
                flattened_labels,
                flattened_class_preds,
                labels=labels,
                average="micro",
                zero_division=np.nan,
            ),
            f"layer_{layer}/{split_name}/f1_weighted": f1_score(
                flattened_labels,
                flattened_class_preds,
                labels=labels,
                average="weighted",
                zero_division=np.nan,
            ),
            f"layer_{layer}/{split_name}/acc": accuracy_score(
                flattened_labels, flattened_class_preds
            ),
            f"layer_{layer}/{split_name}/balanced_acc": balanced_accuracy_score(
                flattened_labels, flattened_class_preds
            ),
            **per_cls_counts,
        }
        metrics.append(layer_metrics)
    return metrics


def train_probes(
    probe_train_states: Sequence[ProbeTrainState],
    config: dict[str, Any],
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    test_data_loader: DataLoader,
    cls_to_tag,
):
    del test_data_loader  # use only when we're ready for final eval
    eval_interval = config["eval_interval"]
    log_interval = config["log_interval"]
    model_name = config["model_name"]
    label = config["label"]
    for epoch in range(config["max_epochs"]):
        iters = len(train_data_loader)
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
                    config=config,
                    cls_to_tag=cls_to_tag,
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
            hidden_states = rearrange(hidden_states, "b l h -> l b h").cuda()
            labels = labels.cuda()
            for probe_train_state in remaining_probe_train_states:
                layer = probe_train_state.layer
                probe = probe_train_state.probe
                probe.train()
                logits = probe(hidden_states[layer])
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer = probe_train_state.optimizer
                optimizer.step()
                optimizer.zero_grad()
                # epoch + probe_train_state.step / iters)

                if probe_train_state.step % log_interval == 0:
                    wandb.log(
                        {
                            f"layer_{probe_train_state.layer}/train/loss": loss,
                            "lr": probe_train_state.scheduler.get_last_lr()[0],
                        },
                        step=probe_train_state.step,
                    )
                probe_train_state.step += 1
        else:
            print(f"Completed epoch {epoch}")
            probe_train_state.scheduler.step()
            continue
        break

    dir_name = os.path.join(label, model_name.replace("/", "_"))
    os.makedirs(dir_name, exist_ok=True)
    for probe_train_state in probe_train_states:
        f = os.path.join(dir_name, f"layer_{probe_train_state.layer}.pt")
        torch.save(
            probe_train_state.best_probe.state_dict(),
            f,
        )
        artifact = wandb.Artifact(
            f"probe_layer_{probe_train_state.layer}", type="model"
        )
        artifact.add_file(f)
        wandb.log_artifact(artifact)
    wandb.finish()
