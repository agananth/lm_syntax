import argparse
from collections.abc import Mapping

import torch
import torch.nn as nn
from stanza.models.constituency import parse_tree, tree_reader
from torch.utils.data import DataLoader
from transformers import AutoConfig

import train
import wandb
import dataset
import utils


def main(parser):
    args = parser.parse_args()

    train_trees = tree_reader.read_treebank(
        "/juice/scr/horatio/data/constituency/en_ptb3-revised_train.mrg"
    )
    all_tags = parse_tree.Tree.get_unique_tags(train_trees)
    num_tags = len(all_tags)

    cls_to_tag: Mapping[int, str] = {i: t for i, t in enumerate(all_tags)}

    model_name = args.model
    model_config = AutoConfig.from_pretrained(model_name)
    num_layers = utils.get_num_layers(model_config)

    config = dict(
        label=args.label,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        model_name=model_name,
        hidden_size=model_config.hidden_size,
        num_layers=num_layers,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        num_classes=num_tags,
    )

    train_data_loader = DataLoader(
        dataset=dataset.WordDataset(
            split_name="train",
            model_name=model_name,
            num_layers=num_layers,
            hidden_size=model_config.hidden_size,
            label_name=args.label,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_data_loader = DataLoader(
        dataset=dataset.WordDataset(
            split_name="dev",
            model_name=model_name,
            num_layers=num_layers,
            hidden_size=model_config.hidden_size,
            label_name=args.label,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_data_loader = DataLoader(
        dataset=dataset.WordDataset(
            split_name="test",
            model_name=model_name,
            num_layers=num_layers,
            hidden_size=model_config.hidden_size,
            label_name=args.label,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    wandb.init(project="Part of Speech Sweeps", name=model_name, config=config)

    probe_train_states = []
    for layer in range(num_layers):
        probe = nn.Linear(
            in_features=model_config.hidden_size, out_features=num_tags, bias=False
        )
        probe.cuda()
        probe_train_states.append(
            train.ProbeTrainState(
                layer=layer,
                probe=probe,
                optimizer=torch.optim.AdamW(probe.parameters(), lr=args.lr),
                early_stopping_metric=train.EarlyStoppingMetric(
                    name="f1_weighted", is_lower_better=False
                ),
                best_metric_value=float("-inf"),
            )
        )

    train.train_probes(
        probe_train_states=probe_train_states,
        config=config,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
        cls_to_tag=cls_to_tag,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="HF base model name"
    )
    parser.add_argument("--label", "-l", type=str, default="pos_cls")
    parser.add_argument("--batch_size", "-b", type=int, default=2048)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-max_epochs", type=int, default=10)
    parser.add_argument("-patience", type=int, default=20)
    parser.add_argument("-eval_interval", type=int, default=10)
    parser.add_argument("-log_interval", type=int, default=10)

    main(parser)
