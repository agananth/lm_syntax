import argparse
import utils
from transformers import AutoConfig, T5ForConditionalGeneration, T5Config
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput
from ReCOGS.utils.compgen import recogs_exact_match
import collections


class COGSDataset(Dataset):
    def __init__(self, split):
        self.srcs = []
        self.tgts = []
        self.categories = []
        for l in open(f"ReCOGS/recogs_v2/{split}.tsv", "r").readlines():
            text, sparse, category = l.split("\t")
            self.srcs.append(text)
            self.tgts.append(sparse)
            self.categories.append(category)

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        return self.srcs[idx], self.tgts[idx], self.categories[idx]


def main(parser):
    args = parser.parse_args()
    model_name = args.model

    tokenizer = utils.get_tokenizer(model_name)
    if args.random_model:
        model = utils.get_model_base_weights(model_name).eval()
        model_state_dict = torch.load(f"{model_name.replace('/', '_')}.pt")
        model.load_state_dict(model_state_dict["model_state_dict"])
    else:
        model = utils.get_model(model_name).eval()
    hidden_size = AutoConfig.from_pretrained(model_name).hidden_size

    checkpoint_path = os.path.join(
        "recogs_checkpoints",
        "random" if args.random_model else "",
        model_name.replace("/", "_"),
        "best_probe.pt",
    )

    print("checkpoint path", checkpoint_path)

    state_dict = torch.load(checkpoint_path)
    print(f"Best epoch: {state_dict['epoch']} validation acc: {state_dict['val_acc']}")
    probe_model_name = "google-t5/t5-small"
    probe_num_decoder_layers = 4
    probe = T5ForConditionalGeneration(
        T5Config.from_pretrained(
            probe_model_name,
            num_layers=0,
            num_decoder_layers=probe_num_decoder_layers,
            hidden_size=hidden_size,
        )
    ).cuda()

    probe.load_state_dict(state_dict["model_state_dict"])
    probe.eval()
    probe_tokenizer = utils.get_tokenizer(probe_model_name)

    gen_correct_counter = collections.Counter()
    gen_total_counter = collections.Counter()
    for split in ("test", "gen"):
        dataset = COGSDataset(split)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        correct = 0
        total = 0
        for src, tgt, categories in tqdm(dataloader):
            with torch.inference_mode():
                src_tokenized = tokenizer(src, return_tensors="pt", padding=True).to(
                    "cuda"
                )
                last_hidden_state = model(
                    **src_tokenized, output_hidden_states=True
                ).last_hidden_state
                encoder_output = BaseModelOutput(
                    last_hidden_state=last_hidden_state,
                    hidden_states=None,
                    attentions=None,
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = probe.generate(
                        attention_mask=src_tokenized.attention_mask,
                        encoder_outputs=encoder_output,
                        max_new_tokens=512,
                    )

                decoded_preds = probe_tokenizer.batch_decode(
                    output, skip_special_tokens=True
                )
                for pred, label, category in zip(decoded_preds, tgt, categories):
                    total += 1
                    if split == "gen":
                        gen_total_counter[category] += 1
                    if recogs_exact_match(label, pred):
                        correct += 1
                        if split == "gen":
                            gen_correct_counter[category] += 1
        print(
            f"{model_name} random={args.random_model} {split} exact match accuracy: {correct/total}"
        )
    print("Gen category accuracies:")
    for category in gen_total_counter:
        print(
            f"{category}: {gen_correct_counter[category]/gen_total_counter[category]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--random_model", action="store_true")
    main(parser)
