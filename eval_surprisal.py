"""Cloned from https://github.com/MurtyShikhar/Pushdown-Layers/blob/main/eval_utils/eval_surprisal.py and kept what is needed to run on base HF models."""

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
import numpy as np
import re
import json
from tqdm import tqdm
import math

import torch
import torch.nn.functional as F
import utils
import os
import run_registry
import csv
import argparse


def eval_math_expr(expr):
    return eval(expr)


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
            target_surprisals = evaluator.get_surprisals_encoder(examples[phen])
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

    def get_surprisals_encoder(self, regions, verbose=False):
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

        input_ids = tokenized.input_ids
        seq_length = input_ids.shape[1]

        masked_input_ids = input_ids.repeat(seq_length, 1)
        extra_mask_tokens = 32
        masked_input_ids = torch.cat(
            (
                masked_input_ids[:, :-1],
                torch.ones(seq_length, extra_mask_tokens, dtype=int).to("cuda"),
                masked_input_ids[:, None, -1],
            ),
            dim=-1,
        )
        mask_token_mask = torch.triu(
            torch.ones(seq_length, seq_length + extra_mask_tokens, dtype=bool)
        )
        assert mask_token_mask.shape == masked_input_ids.shape
        mask_token_mask[torch.arange(seq_length - 1), -1] = False
        masked_input_ids[mask_token_mask] = self.tokenizer.mask_token_id
        masked_attention_mask = torch.ones_like(masked_input_ids)

        with torch.inference_mode():
            all_sent_logprobs = self.lm(
                masked_input_ids,
                attention_mask=masked_attention_mask,
            ).logits
        log_probs = F.log_softmax(all_sent_logprobs, dim=-1)
        token_log_probs = log_probs[
            torch.arange(seq_length), torch.arange(seq_length), input_ids.squeeze()
        ]
        assert token_log_probs.shape == (seq_length,)
        target_surprisals = [
            -1.0 * torch.sum(token_log_probs[st:en], dim=0).item()
            for st, en in all_target_idxs
        ]

        return target_surprisals

    def get_surprisals_encoder_decoder(self, regions, verbose=False):
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

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        seq_length = input_ids.shape[1]

        mask_token = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.additional_special_tokens[0]
        )

        masked_input_ids = input_ids.repeat(seq_length, 1)
        for i in range(seq_length):
            for j in range(i, seq_length):
                if j == seq_length - 1 and i != seq_length - 1:
                    continue
                masked_input_ids[i, j] = self.tokenizer.additional_special_tokens_ids[
                    j - i
                ]
        # print(masked_input_ids)
        # masked_input_ids.fill_diagonal_(mask_token)
        # masked_input_ids[torch.arange(seq_length - 1), torch.arange(1, seq_length)] = (
        #     self.tokenizer.eos_token_id
        # )
        masked_attention_mask = attention_mask.repeat(seq_length, 1)
        # masked_attention_mask = torch.tril(masked_attention_mask, diagonal=1)

        decoder_input_ids = (
            torch.tensor([[self.lm.config.decoder_start_token_id, mask_token]])
            .expand(seq_length, 2)
            .to("cuda")
        )

        with torch.inference_mode():
            all_sent_logprobs = self.lm(
                input_ids=masked_input_ids,
                attention_mask=masked_attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits

        log_probs = F.log_softmax(all_sent_logprobs, dim=-1)
        token_log_probs = log_probs[torch.arange(seq_length), 1, input_ids.squeeze()]

        target_surprisals = [
            -1.0 * torch.sum(token_log_probs[st:en], dim=0).item()
            for st, en in all_target_idxs
        ]

        return target_surprisals


def main(parser):
    args = parser.parse_args()
    if args.encoders:
        auto_model_f = AutoModelForMaskedLM
        model_names = args.encoders
    elif args.decoders:
        auto_model_f = AutoModelForCausalLM
        model_names = args.decoders
    elif args.encoder_decoders:
        auto_model_f = AutoModelForSeq2SeqLM
        model_names = args.encoder_decoders
    else:
        raise ValueError()
    test_suite_dir = "sg_test_suites"
    with open("surprisals_errata.csv", "a", newline="") as csvfile:
        field_names = ["model", "file_name", "item_number", "formula"]
        # writer = csv.DictWriter(csvfile, fieldnames=field_names)
        # writer.writeheader()
        for model_name in model_names:
            results = {}
            lm = auto_model_f.from_pretrained(
                model_name, token="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"
            ).cuda()
            lm.eval()
            tokenizer = utils.get_tokenizer(model_name)

            eval_obj = Evaluator(lm, tokenizer)

            for file_name in os.listdir(test_suite_dir):  # os.listdir(test_suite_dir):
                print("Running", file_name)
                test_suite_parser = TestSuiteParser(
                    os.path.join(test_suite_dir, file_name)
                )
                test_suite_parser.evaluate_all(eval_obj)

                acc = 0.0
                for i, formula in enumerate(test_suite_parser.answers):
                    # print(formula)
                    result = eval_math_expr(formula)
                    acc += result
                    # if not bool(result):
                    #     writer.writerow(
                    #         dict(
                    #             model=model,
                    #             file_name=file_name,
                    #             item_number=i + 1,
                    #             formula=formula,
                    #         )
                    #     )

                acc /= len(test_suite_parser.answers)
                results[file_name] = acc
                print(acc)

            lm = model_name.replace("/", "-")
            output_json = f"surprisals/{lm}.json"
            with open(output_json, "w") as f:
                json.dump(results, f)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", type=str, nargs="+")
    parser.add_argument("--decoders", type=str, nargs="+")
    parser.add_argument("--encoder_decoders", type=str, nargs="+")
    main(parser)
