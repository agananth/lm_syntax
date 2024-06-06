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


api = wandb.Api()


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[(str, str)]]


@dataclasses.dataclass
class SyntaxGymPickle:
    file_name_to_dep_parse: Mapping[str, DepParseDataPickle]


sg_mapping = {
    "center_embed.json": ["plaus"],
    "center_embed_mod.json": ["plaus"],
    "cleft.json": ["np_match", "vp_match"],
    "cleft_modifier.json": ["np_match", "vp_match"],
    "fgd-embed3.json": ["that_no-gap", "what_gap"],
    "fgd-embed4.json": ["that_no-gap", "what_gap"],
    "fgd_hierarchy.json": ["that_nogap", "what_subjgap"],
    "fgd_object.json": ["that_nogap", "what_gap"],
    "fgd_pp.json": ["that_nogap", "what_gap"],
    "fgd_subject.json": ["that_nogap", "what_gap"],
    "mvrr.json": ["unreduced_ambig", "reduced_unambig", "unreduced_unambig"],
    "mvrr_mod.json": ["unreduced_ambig", "reduced_unambig", "unreduced_unambig"],
    "nn-nv-rpl.json": ["nn_unambig", "nv_unambig"],
    "npi_orc_any.json": ["neg_pos", "neg_neg"],
    "npi_orc_ever.json": ["neg_pos", "neg_neg"],
    "npi_src_any.json": ["neg_pos", "neg_neg"],
    "npi_src_ever.json": ["neg_pos", "neg_neg"],
    "npz_ambig.json": ["ambig_comma", "unambig_nocomma", "unambig_comma"],
    "npz_ambig_mod.json": ["ambig_comma", "unambig_nocomma", "unambig_comma"],
    "npz_obj.json": ["no-obj_comma", "obj_no-comma", "obj_comma"],
    "npz_obj_mod.json": ["no-obj_comma", "obj_no-comma", "obj_comma"],
    "number_orc.json": ["match_sing", "match_plural"],
    "number_prep.json": ["match_sing", "match_plural"],
    "number_src.json": ["match_sing", "match_plural"],
    "reflexive_orc_fem.json": ["match_sing", "match_plural"],
    "reflexive_orc_masc.json": ["match_sing", "match_plural"],
    "reflexive_prep_fem.json": ["match_sing", "match_plural"],
    "reflexive_prep_masc.json": ["match_sing", "match_plural"],
    "reflexive_src_fem.json": ["match_sing", "match_plural"],
    "reflexive_src_masc.json": ["match_sing", "match_plural"],
    "subordination.json": ["no-sub_no-matrix", "sub_matrix"],
    "subordination_orc-orc.json": ["no-sub_no-matrix", "sub_matrix"],
    "subordination_pp-pp.json": ["no-sub_no-matrix", "sub_matrix"],
    "subordination_src-src.json": ["no-sub_no-matrix", "sub_matrix"],
}

models_best_layers = {
    "EleutherAI/pythia-70m": 3,
    "EleutherAI/pythia-160m": 6,
    "EleutherAI/pythia-410m": 8,
    "EleutherAI/pythia-1.4b": 10,
    "EleutherAI/pythia-2.8b": 18,
    "tiiuae/falcon-7b": 2,
    "gpt2-large": 15,
    "gpt2-medium": 8,
    "gpt2": 6,
    "EleutherAI/gpt-j-6b": 15,
    "meta-llama/Llama-2-7b-hf": 7,
    "gpt2-xl": 17,
    "mistralai/Mistral-7B-v0.1": 17,
    "google/gemma-2b": 11,
    "microsoft/phi-2": 6,
    "meta-llama/Meta-Llama-3-8B": 14,
    "google/electra-small-generator": 5,
    "google/electra-large-generator": 17,
    "google/electra-base-generator": 6,
    "google-t5/t5-small": 4,
    "google-t5/t5-base": 10,
    "google-t5/t5-large": 12,
    "google-t5/t5-3b": 12,
    "FacebookAI/roberta-large": 8,
    "FacebookAI/roberta-base": 4,
    "EleutherAI/pythia-6.9b": 16,
    "allenai/OLMo-1B-hf": 12,
    "allenai/OLMo-1.7-7B-hf": 4,
}

# Cache SG deps
def cache_data():
    stanza.download("en")
    nlp = stanza.Pipeline("en", tokenize_pretokenized=True)
    mapping = {}
    for file_name, conditions in tqdm(sg_mapping.items()):
        sentences = []
        with open(os.path.join("sg_test_suites", file_name)) as f:
            data = json.load(f)
        for item in data["items"]:
            for cond in item["conditions"]:
                if cond["condition_name"] in conditions:
                    regions = [x["content"] for x in cond["regions"]]
                    regions_cpy = []
                    for region in regions:
                        region = region.lstrip().rstrip()
                        if not region:
                            continue
                        regions_cpy.append(region)
                    sentences.append(" ".join(regions_cpy))

        dev_data = []
        for sentence in sentences:
            doc = nlp(sentence)
            words = []
            heads = []
            relns = []
            for dep_edge in doc.sentences[0].dependencies:
                words.append(dep_edge[2].text)
                heads.append(dep_edge[0].id)
                relns.append(dep_edge[1])
            if not sentence == " ".join(words):
                raise AssertionError(f"sentence: {sentence}, words: {' '.join(words)}")
            dev_data.append(dict(words=words, heads=heads, relns=relns))

        mapping[file_name] = DepParseDataPickle(input_strs=sentences, dev_data=dev_data)
    with open("stanza/sg.pickle", "wb") as writer:
        pickle.dump(SyntaxGymPickle(file_name_to_dep_parse=mapping), writer)



def _read_cache():
    with open("stanza/sg.pickle", "rb") as reader:
        return pickle.load(reader)


def main():
    cache = _read_cache()
    full_results = {}
    for model_name in ["FacebookAI/roberta-base", "allenai/OLMo-1B-hf"]:
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

        probes = []
        for layer in range(num_layers):
            probe = nn.Linear(in_features=hidden_size, out_features=256, bias=False)
            artifact = api.artifact(
                f"ananthag/Head Word Final 2/{model_name.replace('/', '_')}_probe_layer_{layer}:v1"
            )
            path = artifact.download() + f"/layer_{layer}.pt"
            probe.load_state_dict(torch.load(path))
            probe.eval()
            probe.cuda()
            probes.append(probe)

        for file_name, dep_parse_data in tqdm(cache.file_name_to_dep_parse.items()):
            data_loader = DataLoader(
                dataset.SyntaxGymHeadWordDataset(dep_parse_data.dev_data),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda batch: batch,
            )
            flattened_labels = []
            flattened_preds = [[] for _ in range(num_layers)]
            for batch in data_loader:
                words, labels_and_relns = batch[0]
                labels, relns = zip(*labels_and_relns)
                assert len(words) == len(labels)
                input_str = " ".join(words)
                hidden_states = cache_hidden_states.get_word_hidden_states(
                    [input_str], tokenizer, model
                )[0]
                assert len(hidden_states) == num_layers
                labels = torch.tensor([0] + list(labels))
                flattened_labels.extend(labels.tolist())
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
                        flattened_preds[layer].extend(preds.tolist())

            accuracies = []
            for layer, preds in enumerate(flattened_preds):
                accuracy = accuracy_score(flattened_labels, preds)
                accuracies.append(accuracy)
            full_results[file_name] = accuracies[models_best_layers[model_name]]
        
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        with open(f"sg_probe/{model_name.replace("/", "-")}.json", "w") as writer:
            json.dump(
                full_results,
                writer,
            )
    

if __name__ == "__main__":
    # cache_data()
    main()
