"""Cloned from https://github.com/MurtyShikhar/Pushdown-Layers/blob/main/eval_utils/eval_surprisal.py and kept what is needed to run on base HF models."""

from transformers import AutoModelForCausalLM
import numpy as np
import re
import json
from tqdm import tqdm
import math

import torch
import utils
import os
import run_registry


def eval_math_expr(expr):
    try:
        return eval(expr)
    except:
        return math.nan


class TestSuiteParser:
    def __init__(self, test_suite_file):
        self.test_suite_file = test_suite_file
        self.read_test_suite()
        self.answers = [0 for _ in range(len(self.meta_data["data"]))]

    def read_test_suite(self):
        with open(self.test_suite_file, "r") as f:
            data = json.load(f)
        self.meta_data = {
            "formula": data["predictions"][0]["formula"],
            "data": self.get_sents(data),
        }

    def get_sents(self, data):
        all_ex = []
        for item in data["items"]:
            curr_ex = {}
            for cond in item["conditions"]:
                regions = [x["content"] for x in cond["regions"]]
                curr_ex[cond["condition_name"]] = regions
            all_ex.append(curr_ex)
        return all_ex

    def extract_formulas(self, surprisal_dict):
        formula = self.meta_data["formula"]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            positions = set(re.findall(r"\((\d+);%{}%".format(key), formula))
            for position in positions:
                formula = formula.replace(
                    "({};%{}%)".format(position, key),
                    str(surprisal_dict[key][int(position) - 1]),
                )
        ### replace [ with ( and ] with ) to make it a valid math expression

        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        return formula

    def get_example(self, idx):
        return self.meta_data["data"][idx]

    def evaluate_example(self, idx, evaluator, verbose=False):
        examples = self.get_example(idx)
        phen2surprisals = {}
        for phen in examples:
            target_surprisals = evaluator.get_surprisals(examples[phen])
            if verbose:
                print("Regions: {}".format(examples[phen]))
                print(target_surprisals)
            phen2surprisals[phen] = target_surprisals

        extracted_formula = self.extract_formulas(phen2surprisals)
        self.answers[idx] = extracted_formula

    def evaluate_all(self, evaluator):
        for idx in tqdm(range(len(self.meta_data["data"]))):
            self.evaluate_example(idx, evaluator)
        return


class Evaluator:
    def __init__(
        self,
        lm,
        tokenizer,
    ):
        self.lm = lm
        self.tokenizer = tokenizer

    def get_surprisals(self, regions, verbose=False):
        """
        regions: a list of regions which when concatenated with a period and
        processed by the preprocessor, gives a valid input to the language model
        but some regions can be empty, so we need to take care of that
        """

        sent = " ".join([r.lstrip().rstrip() for r in regions if len(r) > 0])
        tokenized = self.tokenizer(sent, return_tensors="pt").to("cuda")

        all_target_idxs = []
        st = 0
        sent_tokens = tokenized.input_ids[0].tolist()
        for idx, region in enumerate(regions):
            region = region.lstrip().rstrip()
            if not region:
                all_target_idxs.append((st, st))
                continue
            word_tokenized = utils.get_tokenized_word(self.tokenizer, region, idx)
            st_curr, en_curr = utils.get_idxs(word_tokenized, sent_tokens, st)
            all_target_idxs.append((st_curr, en_curr))
            st = en_curr
        with torch.inference_mode():
            all_sent_logprobs = self.lm(**tokenized).logits
        targets = torch.cat(
            (
                tokenized.input_ids[0][1:],
                torch.tensor([self.tokenizer.eos_token_id]).to("cuda"),
            ),
            dim=0,
        )
        logprobs = torch.gather(
            torch.nn.functional.log_softmax(all_sent_logprobs[0], dim=1),
            dim=1,
            index=targets.unsqueeze(1),
        ).reshape(-1)
        logprobs = logprobs.roll(1, 0).tolist()
        logprobs[0] = 0.0

        target_surprisals = [
            -1.0 * np.sum(logprobs[st:en], axis=0) for st, en in all_target_idxs
        ]

        return target_surprisals


def main():
    test_suite_dir = "sg_test_suites"
    models = list(run_registry.RUNS.keys())
    results = {}
    for model in models:
        lm = AutoModelForCausalLM.from_pretrained(
            model, token="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"
        ).cuda()
        lm.eval()
        tokenizer = utils.get_tokenizer(model)

        eval_obj = Evaluator(lm, tokenizer)

        for file_name in os.listdir(test_suite_dir):
            test_suite_parser = TestSuiteParser(os.path.join(test_suite_dir, file_name))
            test_suite_parser.evaluate_all(eval_obj)

            acc = 0.0
            for formula in test_suite_parser.answers:
                acc += eval_math_expr(formula)

            acc /= len(test_suite_parser.answers)
            print(model, file_name, acc)
            results[file_name] = acc

        lm = model.replace("/", "-")
        output_json = f"surprisals/{lm}.json"
        with open(output_json, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
