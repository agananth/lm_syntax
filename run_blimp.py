"""Cloned from https://github.com/MurtyShikhar/Pushdown-Layers/blob/main/eval_utils/eval_surprisal.py and kept what is needed to run on base HF models."""

from transformers import AutoModelForCausalLM
import numpy as np
import re
import json
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader
import utils
import os
import run_registry
import csv
from blimp_dataset import BlimpDataset
import utils


class Evaluator:
    def __init__(
        self,
        lm,
        tokenizer,
    ):
        self.lm = lm
        self.tokenizer = tokenizer

    def log_prob(self, input_text_batch: str):
        tokenized = self.tokenizer(
            input_text_batch, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            all_sent_logprobs = self.lm(**tokenized).logits
        batch_size = tokenized.input_ids.shape[0]
        targets = torch.cat(
            (
                tokenized.input_ids[:, 1:],
                torch.tensor([self.tokenizer.eos_token_id])
                .expand(batch_size, 1)
                .to("cuda"),
            ),
            dim=-1,
        )
        logprobs = torch.nn.functional.log_softmax(all_sent_logprobs, dim=-1)
        logprobs = torch.gather(
            logprobs,
            dim=-1,
            index=targets.unsqueeze(-1),
        )
        logprobs = torch.sum(
            logprobs * tokenized.attention_mask.unsqueeze(-1), dim=-1
        ).view(batch_size, -1)
        return logprobs.sum(-1)


def main():
    models = run_registry.RUNS.keys()
    field_names = ["file_name", "accuracy"]
    for model_name in models:
        print(f"Running {model_name}")
        lm = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda()
        tokenizer = utils.get_tokenizer(model_name)
        aggregate_results = {}
        with open(f"blimp_results/{model_name.replace("/", "_")}.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            for file_name in tqdm(sorted(os.listdir("../blimp/data"))):
                    evaluator = Evaluator(lm=lm, tokenizer=tokenizer)
                    dataset = BlimpDataset(file_path=f"../blimp/data/{file_name}")
                    data_loader = DataLoader(
                        dataset=dataset,
                        batch_size=256,
                        shuffle=False,
                    )
                    correct = 0
                    total = 0
                    for batch in data_loader:
                        good_sents, bad_sents = batch
                        good_probs = evaluator.log_prob(good_sents)
                        bad_probs = evaluator.log_prob(bad_sents)
                        correct += torch.sum(good_probs > bad_probs).item()
                        total += len(good_sents)
                    accuracy = correct / total
                    print(file_name, f"Accuracy: {accuracy:.2f}")
                    writer.writerow({"file_name": file_name, "accuracy": accuracy})
                    if dataset.linguistics_term not in aggregate_results:
                        aggregate_results[dataset.linguistics_term] = []
                    aggregate_results[dataset.linguistics_term].append(accuracy)
        with open(f"blimp_results/{model_name.replace('/', '_')}_aggregate.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["linguistics_term", "accuracy"])
            writer.writeheader()
            for term, accuracies in aggregate_results.items():
                writer.writerow({"linguistics_term": term, "accuracy": np.mean(accuracies)})
        del lm, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
