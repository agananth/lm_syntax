from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import os
import argparse
import csv
from blimp_dataset import BlimpDataset
import utils


def _decoder_log_prob_sum(lm, tokenizer, input_text_batch):
    tokenized = tokenizer(input_text_batch, padding=True, return_tensors="pt").to(
        "cuda"
    )
    with torch.inference_mode():
        all_sent_logprobs = lm(**tokenized).logits
    batch_size = tokenized.input_ids.shape[0]
    targets = torch.cat(
        (
            tokenized.input_ids[:, 1:],
            torch.tensor([tokenizer.eos_token_id]).expand(batch_size, 1).to("cuda"),
        ),
        dim=-1,
    )
    logprobs = F.log_softmax(all_sent_logprobs, dim=-1)
    logprobs = torch.gather(
        logprobs,
        dim=-1,
        index=targets.unsqueeze(-1),
    )
    logprobs = torch.sum(
        logprobs * tokenized.attention_mask.unsqueeze(-1), dim=-1
    ).view(batch_size, -1)
    return logprobs.sum(-1)


def _encoder_log_prob_sum(lm, tokenizer, input_text_batch):
    assert len(input_text_batch) == 1
    tokenized = tokenizer(input_text_batch, return_tensors="pt").to("cuda")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    seq_length = input_ids.shape[1]

    # Mask out each token in the input sequence
    masked_input_ids = input_ids.repeat(seq_length, 1)
    masked_attention_mask = attention_mask.repeat(seq_length, 1)
    masked_input_ids.fill_diagonal_(tokenizer.mask_token_id)

    # Get the model's predictions for the batched masked inputs
    with torch.inference_mode():
        logits = lm(masked_input_ids, attention_mask=masked_attention_mask).logits

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[
        torch.arange(seq_length), torch.arange(seq_length), input_ids.squeeze()
    ]
    return token_log_probs.sum()


def _encoder_decoder_log_prob_sum(lm, tokenizer, input_text_batch):
    assert len(input_text_batch) == 1
    tokenized = tokenizer(input_text_batch, return_tensors="pt").to("cuda")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    seq_length = input_ids.shape[1]
    mask_token = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])

    # Mask out each token in the input sequence
    masked_input_ids = input_ids.repeat(seq_length, 1)
    masked_attention_mask = attention_mask.repeat(seq_length, 1)
    masked_input_ids.fill_diagonal_(mask_token)

    decoder_input_ids = (
        torch.tensor([[lm.config.decoder_start_token_id, mask_token]])
        .expand(seq_length, 2)
        .to("cuda")
    )

    # Get the model's predictions for the batched masked inputs
    with torch.inference_mode():
        logits = lm(
            input_ids=masked_input_ids,
            attention_mask=masked_attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[torch.arange(seq_length), 1, input_ids.squeeze()]
    return token_log_probs.sum()


def main(parser):
    args = parser.parse_args()
    if args.encoders:
        log_prob_f = _encoder_log_prob_sum
        auto_model_f = AutoModelForCausalLM
        model_names = args.encoders
        batch_size = 1
    elif args.decoders:
        log_prob_f = _decoder_log_prob_sum
        auto_model_f = AutoModelForMaskedLM
        model_names = args.decoders
        batch_size = 256
    elif args.encoder_decoders:
        log_prob_f = _encoder_decoder_log_prob_sum
        auto_model_f = AutoModelForSeq2SeqLM
        model_names = args.encoder_decoders
        batch_size = 1
    else:
        raise ValueError("Must specify encoders, decoders, or encoder_decoders.")

    field_names = ["file_name", "accuracy"]
    for model_name in model_names:
        print(f"Running {model_name}")
        lm = auto_model_f.from_pretrained(model_name).eval().cuda()
        tokenizer = utils.get_tokenizer(model_name)
        aggregate_results = {}
        with open(
            f"blimp_results/{model_name.replace('/', '_')}.csv", "w", newline=""
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            for file_name in tqdm(sorted(os.listdir("../blimp/data"))):
                dataset = BlimpDataset(file_path=f"../blimp/data/{file_name}")
                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                )
                correct = 0
                total = 0
                for batch in data_loader:
                    good_sents, bad_sents = batch
                    good_probs = log_prob_f(lm, tokenizer, good_sents)
                    bad_probs = log_prob_f(lm, tokenizer, bad_sents)
                    correct += torch.sum(good_probs > bad_probs).item()
                    total += len(good_sents)
                accuracy = correct / total
                print(file_name, f"Accuracy: {accuracy:.2f}")
                writer.writerow({"file_name": file_name, "accuracy": accuracy})
                if dataset.linguistics_term not in aggregate_results:
                    aggregate_results[dataset.linguistics_term] = []
                aggregate_results[dataset.linguistics_term].append(accuracy)
        with open(
            f"blimp_results/{model_name.replace('/', '_')}_aggregate.csv",
            "w",
            newline="",
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["linguistics_term", "accuracy"]
            )
            writer.writeheader()
            for term, accuracies in aggregate_results.items():
                writer.writerow(
                    {"linguistics_term": term, "accuracy": np.mean(accuracies)}
                )
        del lm, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", type=str, nargs="+")
    parser.add_argument("--decoders", type=str, nargs="+")
    parser.add_argument("--encoder_decoders", type=str, nargs="+")
    main(parser)
