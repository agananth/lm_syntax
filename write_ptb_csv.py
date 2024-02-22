import csv
import os

from stanza.models.constituency import tree_reader
from stanza.models.constituency import parse_tree
from collections.abc import Sequence, Mapping
import itertools


def _get_full_input(
    trees: Sequence[parse_tree.Tree], tag_to_cls: Mapping[str, int]
) -> tuple[Sequence[str], Sequence[Sequence[int]]]:
    input_strs, labels = [], []
    for tree in trees:
        sent, pos_classes = [], []
        for preterminal in tree.yield_preterminals():
            tag_str = preterminal.label
            word = preterminal.leaf_labels()[0]
            sent.append(word)
            pos_classes.append(tag_to_cls[tag_str])
        input_strs.append(" ".join(sent))
        labels.append(pos_classes)
    return input_strs, labels


def _height(t: parse_tree.Tree, running_height: int):
    if t.is_leaf():
        return ((t.label, running_height),)
    return list(itertools.chain(*[_height(c, running_height + 1) for c in t.children]))


def main():
    train_trees = tree_reader.read_treebank(
        "/juice/scr/horatio/data/constituency/en_ptb3-revised_train.mrg"
    )
    all_tags = parse_tree.Tree.get_unique_tags(train_trees)
    tag_to_cls: Mapping[str, int] = {t: i for i, t in enumerate(all_tags)}

    dev_trees = tree_reader.read_treebank(
        "/juice/scr/horatio/data/constituency/en_ptb3-revised_dev.mrg"
    )
    test_trees = tree_reader.read_treebank(
        "/juice/scr/horatio/data/constituency/en_ptb3-revised_test.mrg"
    )

    folder_name = "ptb"
    if os.path.exists(folder_name):
        os.rmdir(folder_name)
    os.makedirs(folder_name)
    cache_files = (
        os.path.join(folder_name, "train.csv"),
        os.path.join(folder_name, "dev.csv"),
        os.path.join(folder_name, "test.csv"),
    )

    for cache_file, trees in zip(cache_files, (train_trees, dev_trees, test_trees)):
        with open(cache_file, "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=["word", "pos_tag", "pos_cls", "tree_depth"]
            )
            writer.writeheader()
            for tree in trees:
                for preterminal, (word, depth) in zip(
                    tree.yield_preterminals(), _height(tree, 0)
                ):
                    writer.writerow(
                        dict(
                            word=word,
                            pos_tag=preterminal.label,
                            pos_cls=tag_to_cls[preterminal.label],
                            tree_depth=depth,
                        )
                    )
        print(f"Wrote {cache_file}")


if __name__ == "__main__":
    main()
