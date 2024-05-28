import stanza
from stanza.models.constituency import tree_reader
import dataclasses
from typing import Sequence, Mapping
import pickle
from tqdm import tqdm

_TRAIN_PICKLE_PATH = "stanza/ptb_train.pickle"
_DEV_PICKLE_PATH = "stanza/ptb_dev.pickle"
_TEST_PICKLE_PATH = "stanza/ptb_test.pickle"


@dataclasses.dataclass
class DepParseDataPickle:
    input_strs: Sequence[str]
    dev_data: Sequence[Mapping[(str, str)]]


# def _cache_all_ptb_deps():
#     stanza.download("en")
#     nlp = stanza.Pipeline("en", tokenize_pretokenized=True)
#     train_data = tree_reader.read_treebank(
#         "/juice/scr/horatio/data/constituency/en_ptb3-revised_train.mrg"
#     )
#     val_data = tree_reader.read_treebank(
#         "/juice/scr/horatio/data/constituency/en_ptb3-revised_dev.mrg"
#     )
#     test_data = tree_reader.read_treebank(
#         "/juice/scr/horatio/data/constituency/en_ptb3-revised_test.mrg"
#     )
#     for data, path in ((train_data, _TRAIN_PICKLE_PATH),):
#         dev_data = []
#         for tree in data:
#             input_str = " ".join(tree.leaf_labels())
#             doc = nlp(input_str)
#             words = []
#             heads = []
#             relns = []
#             for dep_edge in doc.sentences[0].dependencies:
#                 words.append(dep_edge[2].text)
#                 heads.append(dep_edge[0].id)
#                 relns.append(dep_edge[1])

#             if not input_str == " ".join(words):
#                 raise AssertionError(
#                     f"input_str: {input_str}, words: {' '.join(words)}"
#                 )
#             else:
#                 dev_data.append(dict(words=words, heads=heads, relns=relns))
#         else:
#             with open(path, "wb") as writer:
#                 pickle.dump(DepParseDataPickle(data, dev_data), writer)
#             print(f"Finished caching {path}")


def _cache_all_ptb_deps():
    stanza.download("en")
    nlp = stanza.Pipeline("en", tokenize_pretokenized=True)
    # train_data = tree_reader.read_treebank("train.tree")
    val_data = tree_reader.read_treebank("dev.tree")
    test_data = tree_reader.read_treebank("test.tree")
    for data, path in (
        # (train_data, _TRAIN_PICKLE_PATH),
        (val_data, _DEV_PICKLE_PATH),
        (test_data, _TEST_PICKLE_PATH),
    ):
        dev_data = []
        for tree in tqdm(data):
            input_str = " ".join(tree.leaf_labels())
            doc = nlp(input_str)
            words = []
            heads = []
            relns = []
            for dep_edge in doc.sentences[0].dependencies:
                words.append(dep_edge[2].text)
                heads.append(dep_edge[0].id)
                relns.append(dep_edge[1])

            if not input_str == " ".join(words):
                raise AssertionError(
                    f"input_str: {input_str}, words: {' '.join(words)}"
                )
            else:
                dev_data.append(dict(words=words, heads=heads, relns=relns))
        else:
            with open(path, "wb") as writer:
                pickle.dump(DepParseDataPickle(data, dev_data), writer)
            print(f"Finished caching {path}")


def _read_pickle(path):
    with open(path, "rb") as reader:
        pickle_obj = pickle.load(reader)
        return pickle_obj


def cached_ptb_train():
    return _read_pickle(_TRAIN_PICKLE_PATH).dev_data


def cached_ptb_train_sentences():
    return _read_pickle(_TRAIN_PICKLE_PATH).input_strs


def cached_ptb_dev():
    return _read_pickle(_DEV_PICKLE_PATH).dev_data


def cached_ptb_dev_sentences():
    return _read_pickle(_DEV_PICKLE_PATH).input_strs


def cached_ptb_test():
    return _read_pickle(_TEST_PICKLE_PATH).dev_data


def cached_ptb_test_sentences():
    return _read_pickle(_TEST_PICKLE_PATH).input_strs


if __name__ == "__main__":
    _cache_all_ptb_deps()
