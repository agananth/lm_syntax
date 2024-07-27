import torch
import json
from torch.utils.data import Dataset, DataLoader
import dataclasses
import argparse
import utils
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoConfig
from tqdm import tqdm
import copy

@dataclasses.dataclass
class Example:
    premise: str
    hypothesis: str
    label: str


class MonliDataset(Dataset):

    def __init__(self, split, separator, eos_token):
        self.data = []
        self.split = split
        with open(f"MoNLI/{split}.jsonl") as f:
            for line in f:
                example = json.loads(line)
                example = Example(
                    premise=example["sentence1"],
                    hypothesis=example["sentence2"],
                    label=example["gold_label"],
                )
                self.data.append(example)
        self.separator = separator
        self.eos_token = eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        formatted_input = (
            f"Premise: {example.premise}{self.separator}Hypothesis: {example.hypothesis}{self.separator}Label:"
        )
        if self.split == "train":
            formatted_input += " " + example.label + self.eos_token
        return formatted_input, example.label


class Probe(GPT2LMHeadModel):

    def __init__(self, config, model_name):
        super().__init__(config)
        self.transformer = utils.get_model(model_name)
        for param in self.transformer.parameters():
            param.requires_grad = False
        hidden_size = self.transformer.config.hidden_size
        vocab_size = self.transformer.config.vocab_size
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GeLU(),
            nn.Linear(hidden_size, vocab_size, bias=False),
        )
        for param in self.lm_head.parameters():
            param.requires_grad = True


def main(args):
    model_name = args.model
    tokenizer = utils.get_tokenizer(model_name)
    eval_tokenizer = utils.get_tokenizer(model_name)
    eval_tokenizer.padding_side = "left"

    train_dataset = MonliDataset("train", separator=tokenizer.cls_token or "\n", eos_token=tokenizer.eos_token)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = MonliDataset("dev", separator=eval_tokenizer.cls_token or "\n", eos_token=eval_tokenizer.eos_token)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    probe = Probe(config=AutoConfig.from_pretrained(model_name), model_name=model_name)
    probe.train()
    probe.cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    
    neutral_token_length = len(utils.get_tokenized_word(tokenizer, "neutral", index=-1))
    entailment_token_length = len(utils.get_tokenized_word(tokenizer, "entailment", index=-1))

    for _ in range(20):
        probe.train()
        for batch in tqdm(train_dataloader):
            input_batch, labels = batch
            input_tokens = tokenizer(input_batch, padding=True, return_tensors="pt").to("cuda")
            input_ids = input_tokens.input_ids

            # Set everything outside neutral/entailment and EOS to -100
            indices = input_tokens.attention_mask.sum(dim=-1)
            original_indices = indices.clone()
            indices = indices - 1
            for i, label in enumerate(labels):
                if label == "neutral":
                    indices[i] -= neutral_token_length
                elif label == "entailment":
                    indices[i] -= entailment_token_length
                else:
                    raise ValueError(f"Invalid label: {label}")
            expanded = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).cuda()
            mask = torch.logical_or(expanded < indices.unsqueeze(1), expanded >= original_indices.unsqueeze(1))
            labels = torch.masked_fill(input_ids, mask, -100)
            
            output = probe(**input_tokens, labels=labels)
            loss = output.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        probe.eval()
        total = 0
        correct = 0
        for batch in dev_dataloader:
            input_batch, labels = batch
            input_tokens = eval_tokenizer(input_batch, padding=True, return_tensors="pt").to("cuda")
            
            output = probe.generate(**input_tokens, max_new_tokens=4, eos_token_id=eval_tokenizer.eos_token_id)
            generated_tokens = output[:, input_tokens.input_ids.size(1):]
            decoded_preds = eval_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for pred, label in zip(decoded_preds, labels):
                if pred.lstrip() == label:
                    correct += 1
                total += 1
        print(f"Accuracy: {correct / total:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    main(parser.parse_args())
