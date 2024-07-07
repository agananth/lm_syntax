import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from stanza.models.constituency import tree_reader
import dataclasses
from typing import Sequence, Mapping
import pickle
import stanza
import utils
import run_registry
from torch.utils.data import DataLoader
import dataset
import cache_hidden_states
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
import blimp_dataset
from scipy.spatial import distance


_SUBJECT_VERB_AGREEMENT_FILES = (
    "distractor_agreement_relational_noun.jsonl",
    "distractor_agreement_relative_clause.jsonl",
)

_FILLER_GAP_FILES = (
    "wh_vs_that_no_gap_long_distance.jsonl",
)

_FULL_WORD_FILTER = {
    "have",
    "has",
    "had",
    "having",
    "being",
    "be",
    "is",
    "am",
    "are",
    "was",
    "will",
    "were",
    "do",
    "does",
    "must",
    "should",
    "would",
    "can",
    "might",
    "could",
    "shall",
    "may",
    "ought",
}
_ENDS_WITH_FILTER = {"'d", "'s", "'m", "'re", "'ll", "’s", "n't"}


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[(str, str)]]


@dataclasses.dataclass
class BlimpPickle:
    file_name_to_dep_parse: Mapping[str, DepParseDataPickle]


def cache_data():
    stanza.download("en")
    nlp = stanza.Pipeline("en")
    mapping = {}
    for file_name in _FILLER_GAP_FILES:
        print(file_name)
        dataset = blimp_dataset.BlimpDataset(f"../blimp/data/{file_name}")
        dev_data = []
        valid_sentences = []
        for good_sent, _ in tqdm(dataset):
            doc = nlp(good_sent)
            words = []
            heads = []
            relns = []
            for dep_edge in doc.sentences[0].dependencies:
                words.append(dep_edge[2].text)
                heads.append(dep_edge[0].id)
                relns.append(dep_edge[1])
            dev_data.append(dict(words=words, heads=heads, relns=relns))
            valid_sentences.append(" ".join(words))

        mapping[file_name] = DepParseDataPickle(
            input_strs=valid_sentences, dev_data=dev_data
        )
    with open("stanza/blimp_mark.pickle", "wb") as writer:
        pickle.dump(BlimpPickle(file_name_to_dep_parse=mapping), writer)


def _read_cache():
    with open("stanza/blimp_mark.pickle", "rb") as reader:
        return pickle.load(reader)


def main():
    cache = _read_cache()
    for model_name in run_registry.RUNS.keys():
        if "70m" in model_name or "160m" in model_name:
            continue
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

        probes = [
            utils.load_probe_for_layer(
                model_name, layer_idx=layer, hidden_size=hidden_size
            )
            for layer in range(num_layers)
        ]

        full_results = {}
        for file_name, dep_parse_data in tqdm(cache.file_name_to_dep_parse.items()):
            head_word_dataset = dataset.SyntaxGymHeadWordDataset(dep_parse_data.dev_data)
            data_loader = DataLoader(
                head_word_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda batch: batch,
            )
            all_labels = []
            all_rels = []
            all_preds = [[] for _ in range(num_layers)]
            for batch in data_loader:
                words, labels_and_relns = batch[0]
                labels, relns = zip(*labels_and_relns)
                assert len(words) == len(labels)
                input_str = " ".join(words)
                hidden_states = cache_hidden_states.get_word_hidden_states(
                    [input_str], tokenizer, model
                )[0]
                assert len(hidden_states) == num_layers
                all_rels.append(["ROOT"] + list(relns))
                labels = torch.tensor([0] + list(labels))
                all_labels.append(labels.tolist())
                labels = labels.cuda()
                for layer, probe in enumerate(probes):
                    layer_hidden_states = torch.cat(
                        (root_hidden_state[layer], hidden_states[layer]), dim=0
                    ).cuda()
                    assert layer_hidden_states.shape == (
                        len(words) + 1,
                        hidden_size,
                    )
                    with torch.inference_mode():
                        logits = probe(layer_hidden_states)
                        distances = -torch.cdist(logits, logits)
                        # Exclude root from masking
                        mask = F.pad(
                            torch.eye(distances.shape[0] - 1), (1, 0, 1, 0)
                        ).bool()
                        distances[mask] = float("-inf")
                        preds = distances.softmax(dim=-1).argmax(dim=-1)
                        all_preds[layer].append(preds.tolist())

            assert (
                len(all_rels) == len(all_labels) == len(all_preds[0])
            )
            # blimp_dataset_prefixes = blimp_dataset.BlimpDatasetWithWords(f"../blimp/data/{file_name}")
            # assert len(all_rels) == len(blimp_dataset_prefixes)
            layer_results = {}
            for layer, preds in enumerate(all_preds):
                predictions_correct_invalid = []
                # for sent_preds, sent_rels, sent_labels, prefix_of_interest, words in zip(preds, all_rels, all_labels, blimp_dataset_prefixes, head_word_dataset.words):
                for sent_preds, sent_rels, sent_labels, words in zip(preds, all_rels, all_labels, head_word_dataset.words):
                    # prefix_of_interest = prefix_of_interest.split(" ")[0]
                    if 'mark' not in sent_rels:
                        predictions_correct_invalid.append(-1)
                        continue
                    # if prefix_of_interest not in words:
                        # predictions_correct_invalid.append(-1)
                        # continue
                    # index = words.index(prefix_of_interest) + 1
                    for pred, rel, label in zip(sent_preds, sent_rels, sent_labels):
                        # if label == index:
                        if rel == "mark":
                            predictions_correct_invalid.append(int(pred == label))
                            break
                    # else:
                    #     predictions_correct_invalid.append(-1)
                assert predictions_correct_invalid.count(-1) == 7
                layer_results[layer] = predictions_correct_invalid
                            
            full_results[file_name] = layer_results

        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        test_suite_results = {}
        for file_name, layer_results in full_results.items():
            with open(f"blimp_results/{model_name.replace('/', '_')}_{file_name[:-1]}", "r") as reader:
                data = json.load(reader)["correct"]

            probe_accuracies = []
            end_accuracies = []
            jaccards = []
            hammings = []
            N = None
            for layer, results in layer_results.items():
                probe_vals = []
                end_vals = []
                assert len(results) == len(data) == 1000
                for probe_val, end_val in zip(results, data):
                    if probe_val == -1:
                        continue
                    probe_vals.append(probe_val)
                    end_vals.append(int(end_val))
                probe_accuracies.append(np.sum(probe_vals) / len(probe_vals))
                end_accuracies.append(np.sum(end_vals) / len(end_vals))
                jaccards.append(distance.jaccard(probe_vals, end_vals))
                hammings.append(distance.hamming(probe_vals, end_vals))
                N = len(probe_vals)
            argmax = np.argmax(probe_accuracies)
            max_probe_acc = probe_accuracies[argmax]
            max_end_acc = end_accuracies[argmax]
            max_jaccard = jaccards[argmax]
            max_hamming = hammings[argmax]
            test_suite_results[file_name] = dict(
                probe_acc=max_probe_acc,
                blimp_acc=max_end_acc,
                jaccard=max_jaccard,
                hamming=max_hamming,
                N=N,
            )

        with open(f"blimp_probe/{model_name.replace("/", "-")}_mark.json", "w") as writer:
            json.dump(
                test_suite_results,
                writer,
            )


if __name__ == "__main__":
    # cache_data()
    main()
