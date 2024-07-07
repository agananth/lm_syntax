import csv
import json


def main():
    file_names = ["distractor_agreement_relative_clause.jsonl"]
    with open("blimp_probe/gpt2.json", "r") as f:
        data = json.load(f)
    for file_name in file_names:
        probe_preds = data[file_name]["best_layer_results"]
        with open(f"blimp_results/gpt2_{file_name}.json", "r") as f:
            blimp_preds = json.load(f)["correct"]
        blimp_full_data = None
        with open(f"../blimp/data/{file_name}", "r") as f:
            blimp_full_data = [json.loads(l) for l in f.readlines()]
        assert len(probe_preds) == len(blimp_preds) == len(blimp_full_data), (
            len(probe_preds),
            len(blimp_preds),
            len(blimp_full_data),
        )
        both_correct = 0
        probe_right_behavior_wrong = 0
        probe_wrong_behavior_right = 0
        for i, (probe_pred, blimp_pred, blimp_data) in enumerate(
            zip(probe_preds, blimp_preds, blimp_full_data)
        ):
            if not probe_pred:
                continue
            all_words = blimp_data["sentence_good"].split(" ")
            one_word_prefix = blimp_data["one_prefix_word_good"].split(" ")[0]
            if one_word_prefix not in probe_pred:
                continue
            if blimp_pred:
                if probe_pred[one_word_prefix]:
                    both_correct += 1
                else:
                    probe_wrong_behavior_right += 1
            elif probe_pred[one_word_prefix]:
                probe_right_behavior_wrong += 1
        print(
            f"Both correct: {both_correct}, probe right behavior wrong: {probe_right_behavior_wrong}, probe wrong behavior right: {probe_wrong_behavior_right}"
        )


if __name__ == "__main__":
    main()
