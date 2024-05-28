from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import utils
from ReCOGS.utils.compgen import recogs_exact_match


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
    with torch.inference_mode():
        for src, tgt in val_dataloader:
            # src_inputs = model_tokenizer(src, return_tensors="pt", padding=True).to(
            #     "cuda"
            # )
            # encoder_output = model(**src_inputs, output_hidden_states=True)
            src_inputs = probe_tokenizer(src, return_tensors="pt", padding=True).to(
                "cuda"
            )

            output = probe.generate(
                # attention_mask=src_inputs.attention_mask,
                # encoder_outputs=encoder_output,
                **src_inputs,
                max_length=128,
                decoder_start_token_id=probe_tokenizer.cls_token_id,
                pad_token_id=probe_tokenizer.pad_token_id,
                eos_token_id=probe_tokenizer.sep_token_id,
                # vocab_size=probe.config.decoder.vocab_size,
            )
            decoded_preds = probe_tokenizer.batch_decode(
                output, skip_special_tokens=True
            )
            for pred, label in zip(decoded_preds, tgt):
                total += 1
                if total < 5:
                    print(pred)
                    print(label)
                if recogs_exact_match(label, pred):
                    correct += 1
    print("Accuracy: ", correct / total)


def main():
    dataset = COGSDataset("train")
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(COGSDataset("dev"), batch_size=256, shuffle=False)
    tokenizer = utils.get_tokenizer("gpt2-xl")
    model = utils.get_model_base_weights("gpt2-xl").eval()

    # probe = T5ForConditionalGeneration(
    #     T5Config.from_pretrained(
    #         "google-t5/t5-small",
    #         num_layers=0,
    #         num_decoder_layers=4,
    #         hidden_size=model.config.hidden_size,
    #     )
    # ).cuda()
    probe = EncoderDecoderModel(
        config=EncoderDecoderConfig.from_encoder_decoder_configs(
            BertConfig(
                num_hidden_layers=4,
                hidden_size=300,
                num_attention_heads=4,
            ),
            BertConfig(
                num_hidden_layers=2,
                hidden_size=300,
                num_attention_heads=4,
            ),
        )
    ).cuda()
    probe_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
    probe.config.decoder_start_token_id = probe_tokenizer.cls_token_id
    probe.config.pad_token_id = probe_tokenizer.pad_token_id
    probe.config.vocab_size = probe.config.decoder.vocab_size
    probe.config.eos_token_id = probe_tokenizer.sep_token_id
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * len(train_dataloader),
        num_training_steps=len(train_dataloader) * 3,
    )

    step = 0
    for _ in range(3):
        for src, tgt in train_dataloader:
            if step and step % 500 == 0:
                _run_eval(probe, probe_tokenizer, model, tokenizer, val_dataloader)
            probe.train()

            # src_inputs = tokenizer(src, return_tensors="pt", padding=True).to("cuda")
            src_inputs = probe_tokenizer(src, return_tensors="pt", padding=True).to(
                "cuda"
            )
            # with torch.no_grad():
            #     src_outputs = model(**src_inputs, output_hidden_states=True)
            # src_hidden_states = src_outputs.hidden_states[
            #     model.config.num_hidden_layers // 2
            # ]

            tgt_inputs = probe_tokenizer(tgt, return_tensors="pt", padding=True).to(
                "cuda"
            )

            tgt_inputs.input_ids.masked_fill_(
                tgt_inputs.input_ids == probe_tokenizer.pad_token_id, -100
            )
            # print("TGT INPUTS", tgt_inputs)

            output = probe(
                # attention_mask=src_inputs.attention_mask,
                # encoder_outputs=src_outputs,
                **src_inputs,
                labels=tgt_inputs.input_ids,
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 20 == 0:
                print("Loss: ", loss.item())
            step += 1


if __name__ == "__main__":
    main()
