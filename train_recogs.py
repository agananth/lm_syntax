from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertLMHeadModel,
    BertConfig,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    GPT2Config,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import utils
from ReCOGS.utils.compgen import recogs_exact_match
from tqdm import tqdm
import numpy as np


class COGSDataset(Dataset):
    def __init__(self, split):
        self.srcs = []
        self.tgts = []
        for l in open(f"ReCOGS/recogs_v2/{split}.tsv", "r").readlines():
            text, sparse, _ = l.split("\t")
            self.srcs.append(text)
            self.tgts.append(sparse)

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        return self.srcs[idx], self.tgts[idx]


def _run_eval(probe, probe_tokenizer, model, model_tokenizer, val_dataloader):
    probe.eval()
    correct = 0
    total = 0
    losses = []
    with torch.inference_mode():
        for src, tgt in val_dataloader:
            src_inputs = model_tokenizer(src, return_tensors="pt", padding=True).to(
                "cuda"
            )
            encoder_output = model(**src_inputs, output_hidden_states=True)
            # src_inputs = probe_tokenizer(src, return_tensors="pt", padding=True).to(
            #     "cuda"
            # )

            tgt_inputs = probe_tokenizer(tgt, return_tensors="pt", padding=True).to(
                "cuda"
            )
            labels = tgt_inputs.input_ids.masked_fill(
                tgt_inputs.input_ids == probe_tokenizer.pad_token_id, -100
            )
            # output = probe(
            #     **tgt_inputs,
            #     labels=labels,
            #     encoder_attention_mask=src_inputs.attention_mask,
            #     encoder_hidden_states=encoder_output.last_hidden_state,
            #     # **src_inputs,
            # )
            # losses.append(output.loss.item())
            # prediction = output.logits.argmax(dim=-1)
            # decoded_preds = probe_tokenizer.batch_decode(
            # prediction, skip_special_tokens=True
            # )
            # batch_size = len(tgt)
            # print(batch_size)
            output = probe.generate(
                # attention_mask=src_inputs.attention_mask,
                # encoder_outputs=encoder_output,
                # input_ids=torch.ones((batch_size, 1), dtype=torch.long, device="cuda")
                # * probe.config.decoder_start_token_id,
                encoder_attention_mask=src_inputs.attention_mask,
                encoder_hidden_states=encoder_output.last_hidden_state,
                # **src_inputs,
                max_length=128,
                bos_token_id=probe_tokenizer.cls_token_id,
                decoder_start_token_id=probe_tokenizer.cls_token_id,
                pad_token_id=probe_tokenizer.pad_token_id,
                eos_token_id=probe.config.eos_token_id,
                # vocab_size=probe.config.vocab_size,
            )
            decoded_preds = probe_tokenizer.batch_decode(
                output, skip_special_tokens=True
            )
            for pred, label in zip(decoded_preds, tgt):
                total += 1
                if total < 2:
                    print(pred)
                    print(label)
                if recogs_exact_match(label, pred):
                    correct += 1
    print("Avg loss", np.mean(losses), "Accuracy: ", correct / total)


def main():
    dataset = COGSDataset("train")
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(COGSDataset("dev"), batch_size=256, shuffle=False)
    tokenizer = utils.get_tokenizer("gpt2")

    # probe = T5ForConditionalGeneration(
    #     T5Config.from_pretrained(
    #         "google-t5/t5-small",
    #         num_layers=0,
    #         num_decoder_layers=4,
    #         hidden_size=model.config.hidden_size,
    #     )
    # ).cuda()
    model = utils.get_model_base_weights("gpt2").eval()
    probe = BertLMHeadModel(
        config=BertConfig(
            num_hidden_layers=4,
            hidden_size=model.config.hidden_size,
            is_decoder=True,
            add_cross_attention=True,
        )
    ).cuda()
    learning_rate = 1e-4
    num_epochs = 20

    probe_tokenizer = utils.get_tokenizer("google-bert/bert-base-cased")
    probe.config.decoder_start_token_id = probe_tokenizer.cls_token_id
    # probe.config.pad_token_id = probe_tokenizer.pad_token_id
    # probe.config.vocab_size = probe.config.decoder.vocab_size
    probe.config.eos_token_id = probe_tokenizer.sep_token_id
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)
    ttotal = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * ttotal,
        num_training_steps=ttotal,
    )

    step = 0
    for _ in range(num_epochs):
        batch_iterator = tqdm(train_dataloader)
        for src, tgt in batch_iterator:
            if step % 1000 == 0:
                _run_eval(probe, probe_tokenizer, model, tokenizer, val_dataloader)
            probe.train()

            src_inputs = tokenizer(src, return_tensors="pt", padding=True).to("cuda")

            # src_inputs = probe_tokenizer(src, return_tensors="pt", padding=True).to(
            #     "cuda"
            # )
            with torch.no_grad():
                src_outputs = model(**src_inputs, output_hidden_states=True)
            # src_hidden_states = src_outputs.hidden_states[
            #     model.config.num_hidden_layers // 2
            # ]

            tgt_inputs = probe_tokenizer(tgt, return_tensors="pt", padding=True).to(
                "cuda"
            )

            labels = tgt_inputs.input_ids.masked_fill(
                tgt_inputs.input_ids == probe_tokenizer.pad_token_id, -100
            )
            # print(src_outputs.last_hidden_state.shape, src_inputs.input_ids.shape)
            # raise ValueError()

            output = probe(
                **tgt_inputs,
                labels=labels,
                encoder_attention_mask=src_inputs.attention_mask,
                encoder_hidden_states=src_outputs.last_hidden_state,
                # **src_inputs,
            )
            first_output = output.logits.argmax(-1)

            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # if step % 20 == 0:
            #     print("Loss: ", loss.item())
            batch_iterator.set_postfix({"loss": round(loss.item(), 2)})
            step += 1


if __name__ == "__main__":
    main()
