import argparse
import os
from collections.abc import Sequence, Mapping

import numpy as np
import torch
from tqdm import tqdm
import dataclasses

from transformers import set_seed
import utils
import stanza_cache
from stanza.models.constituency import parse_tree


set_seed(42)


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[(str, str)]]


def get_idxs(phrase_tokens, sent_tokens, st):
    while st < len(sent_tokens):
        en = st + len(phrase_tokens)
        if sent_tokens[st:en] == phrase_tokens:
            return (st, en)
        st += 1
    ### should not get to this point!
    raise AssertionError(
        f"Could not find indices of phrase tokens {phrase_tokens} in sentence tokens {sent_tokens}"
    )


def _get_pre_tokenized_info(tokenizer, input_str: str):
    """
    e.g.
        input_str: The man is eating bananas
        [The, man, is, eating, bananas]
        [The, man, is, eat##, ##ing, bananas]
        [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6)]
    """
    sent_tokens = tokenizer(input_str).input_ids
    words = input_str.split(" ")
    idxs = []
    # go in order.
    st = 0
    for i, word in enumerate(words):
        # GPT2 tokenizer treats spaces like parts of the tokens
        word_tokenized = utils.get_tokenized_word(tokenizer, word=word, index=i)
        st_curr, en_curr = get_idxs(word_tokenized, sent_tokens, st)
        idxs.append((st_curr, en_curr))
        st = en_curr
    return idxs


def get_word_hidden_states(input_batch, tokenizer, model):
    token_input = tokenizer(input_batch, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        token_hidden_states_batch = model(
            **token_input, output_hidden_states=True
        ).hidden_states
    num_layers = len(token_hidden_states_batch) - 1
    # token_hidden_states_batch: (num_layers, batch_size, num_tokens, hidden_size)
    word_level_hs_batch = []
    # output shape is (batch_size, num_layers, num_words, hidden_size)
    for i, input_s in enumerate(input_batch):
        idxs = _get_pre_tokenized_info(tokenizer, input_s)
        num_words = len(input_s.split(" "))
        assert num_words == len(idxs)
        word_hs_combined_from_tokens = []
        for layer in range(num_layers):
            word_level_hidden_states = torch.zeros(
                (num_words, model.config.hidden_size)
            )
            remove_token_embeddings = (
                token_hidden_states_batch[layer + 1][i]
                - token_hidden_states_batch[0][i]
            )
            for word_idx, (st, en) in enumerate(idxs):
                word_level_hidden_states[word_idx] = remove_token_embeddings[
                    st:en
                ].mean(dim=0)
            word_hs_combined_from_tokens.append(word_level_hidden_states)

        word_level_hs_batch.append(word_hs_combined_from_tokens)

    return word_level_hs_batch


def _get_sentences_from_trees(trees: Sequence[parse_tree.Tree]) -> Sequence[str]:
    return [" ".join(tree.leaf_labels()) for tree in trees]


def main(parser):
    args = parser.parse_args()

    train_input_strs = _get_sentences_from_trees(
        stanza_cache.cached_ptb_train_sentences()
    )
    dev_input_strs = _get_sentences_from_trees(stanza_cache.cached_ptb_dev_sentences())
    test_input_strs = _get_sentences_from_trees(
        stanza_cache.cached_ptb_test_sentences()
    )

    batch_size = args.batch_size
    model_name = args.model
    tokenizer = utils.get_tokenizer(model_name)
    if args.random_model:
        print("Using random model")
        model = utils.get_model_base_weights(model_name)
    else:
        print("Using pretrained model")
        model = utils.get_model(model_name)
    model.eval()
    num_layers = utils.get_num_layers(model.config)
    hidden_size = model.config.hidden_size

    folder_path = os.path.join(
        "/scr/biggest/ananthag",
        "hidden_states",
        "random" if args.random_model else "",
        model_name.replace("/", "_"),
    )
    if os.path.exists(folder_path):
        os.rmdir(folder_path)
    os.makedirs(folder_path)
    cache_files = (
        os.path.join(folder_path, "train.dat"),
        os.path.join(folder_path, "dev.dat"),
        os.path.join(folder_path, "test.dat"),
    )

    for cache_file, input_strs in zip(
        cache_files, (train_input_strs, dev_input_strs, test_input_strs)
    ):
        open(cache_file, "w").close()
        offset = 0
        for i in tqdm(range(0, len(input_strs), batch_size)):
            input_batch = input_strs[i : i + batch_size]

            word_level_hidden_states_batch = get_word_hidden_states(
                input_batch=input_batch,
                tokenizer=tokenizer,
                model=model,
            )

            for word_level_hidden_states, input_str in zip(
                word_level_hidden_states_batch, input_batch
            ):
                to_cache = torch.cat(word_level_hidden_states, dim=1)
                num_words = len(input_str.split(" "))
                assert to_cache.shape == (num_words, num_layers * hidden_size)
                cache = np.memmap(
                    cache_file,
                    mode="r+",
                    shape=to_cache.shape,
                    dtype=np.float32,
                    offset=offset,
                )
                cache[:] = to_cache.numpy()
                offset += to_cache.element_size() * to_cache.nelement()

        print(f"Finished caching {cache_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="HF base model name"
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--random_model", action="store_true")

    main(parser)
