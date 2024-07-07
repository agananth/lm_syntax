from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    BertLMHeadModel,
    BertConfig,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    GPT2Config,
    T5ForConditionalGeneration,
    T5Config,
    AutoConfig,
)
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import utils
from ReCOGS.utils.compgen import recogs_exact_match
from tqdm import tqdm
import numpy as np
import wandb
import argparse
import os
import gc


class COGSDataset(Dataset):
    def __init__(self, split, tokenizer, hidden_size, model_name, is_random_model):
        self.srcs = []
        self.tgts = []
        self.start_indices = []
        total_words = 0
        for l in open(f"ReCOGS/recogs_v2/{split}.tsv", "r").readlines():
            text, sparse, _ = l.split("\t")
            self.srcs.append(text)
            self.tgts.append(sparse)
            self.start_indices.append(total_words)
            tokenized_text = tokenizer(text, return_tensors="pt")
            total_words += tokenized_text.input_ids.shape[1]

        self.hidden_state_cache = np.memmap(
            os.path.join(
                "/scr/biggest/ananthag",
                # "hidden_states",
                "recogs",
                "random" if is_random_model else "",
                model_name.replace("/", "_"),
                f"{split}.dat",
            ),
            dtype=np.float32,
            mode="r",
            shape=(total_words, hidden_size),
        )

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        if idx < 0:
            idx = len(self.start_indices) + idx
        end_index = (
            self.start_indices[idx + 1]
            if idx + 1 < len(self.start_indices)
            else self.hidden_state_cache.shape[0]
        )
        return (
            self.hidden_state_cache[start_index:end_index],
            self.srcs[idx],
            self.tgts[idx],
        )


def custom_collate_fn(batch):
    max_dim_0 = 0
    hidden_size = None
    for item in batch:
        max_dim_0 = max(max_dim_0, item[0].shape[0])
        hidden_size = item[0].shape[1]
    assert max_dim_0
    srcs = []
    tgts = []
    padded_input = torch.zeros((len(batch), max_dim_0, hidden_size))
    for i, item in enumerate(batch):
        padded_input[i, : item[0].shape[0]] = torch.from_numpy(item[0])
        srcs.append(item[1])
        tgts.append(item[2])
    return padded_input, tuple(srcs), tuple(tgts)


def _run_eval(probe, probe_tokenizer, model_tokenizer, val_dataloader):
    probe.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for hidden_states, src, tgt in tqdm(val_dataloader):
            src_inputs = model_tokenizer(src, return_tensors="pt", padding=True)
            # encoder_output = model(**src_inputs, output_hidden_states=True)
            assert torch.eq(
                src_inputs.attention_mask.unsqueeze(-1) * hidden_states, hidden_states
            ).all()

            attention_mask_cuda = src_inputs.attention_mask.to("cuda")
            last_hidden_states_cuda = hidden_states.to("cuda")
            encoder_output = BaseModelOutput(
                last_hidden_state=last_hidden_states_cuda,
                hidden_states=None,
                attentions=None,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = probe.generate(
                    attention_mask=attention_mask_cuda,
                    encoder_outputs=encoder_output,
                    max_new_tokens=512,
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

            del (
                output,
                last_hidden_states_cuda,
                attention_mask_cuda,
            )

    val_accuracy = correct / total
    print("Val Accuracy: ", val_accuracy)
    wandb.log({"val_accuracy": val_accuracy})
    return val_accuracy


def main(parser):
    args = parser.parse_args()

    model_name = args.model
    tokenizer = utils.get_tokenizer(model_name)
    hidden_size = AutoConfig.from_pretrained(model_name).hidden_size

    dataset = COGSDataset(
        "train", tokenizer, hidden_size, model_name, args.random_model
    )
    train_batch_size = 16
    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        COGSDataset("dev", tokenizer, hidden_size, model_name, args.random_model),
        batch_size=256,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    probe_model_name = "google-t5/t5-small"
    probe_num_decoder_layers = 4
    probe_tokenizer = utils.get_tokenizer(probe_model_name)
    probe = T5ForConditionalGeneration(
        T5Config.from_pretrained(
            probe_model_name,
            num_layers=0,
            num_decoder_layers=probe_num_decoder_layers,
            hidden_size=hidden_size,
        )
    ).cuda()

    learning_rate = args.learning_rate
    num_epochs = args.epochs
    best_val_acc = 0
    wandb.init(
        project="recogs",
        name=model_name + "_random" if args.random_model else model_name,
        config=dict(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            probe_model_name=probe_model_name,
            probe_num_decoder_layers=probe_num_decoder_layers,
            train_batch_size=train_batch_size,
            dtype="bfloat16",
            num_decode_steps=512,
        ),
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)
    ttotal = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * ttotal,
        num_training_steps=ttotal,
    )

    checkpoint_dir = os.path.join(
        "recogs_checkpoints",
        "random" if args.random_model else "",
        model_name.replace("/", "_"),
    )
    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )
    step = 0
    log_interval = 5000
    best_epoch = -1
    patience = 0
    early_stopping = False
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        run_val_already = False
        batch_iterator = tqdm(train_dataloader)
        for hidden_states, src, tgt in batch_iterator:
            if epoch and not run_val_already and (epoch % 10 == 0 or epoch > 249):
                val_acc = _run_eval(
                    probe=probe,
                    probe_tokenizer=probe_tokenizer,
                    model_tokenizer=tokenizer,
                    val_dataloader=val_dataloader,
                )
                run_val_already = True
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_model_path = os.path.join(checkpoint_dir, "best_probe.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": probe.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": val_acc,
                        },
                        best_model_path,
                    )
                    wandb.save(best_model_path)
                elif epoch >= 200:
                    patience += 1
                    if patience >= 10:
                        print("Early stopping at epoch: ", epoch)
                        early_stopping = True
                        break

            probe.train()

            src_inputs = tokenizer(src, return_tensors="pt", padding=True)

            # with torch.no_grad():
            #     src_outputs = model(**src_inputs, output_hidden_states=True)

            # assert torch.eq(
            #     src_inputs.attention_mask.unsqueeze(-1) * hidden_states, hidden_states
            # ).all()

            tgt_inputs = probe_tokenizer(tgt, return_tensors="pt", padding=True).to(
                "cuda"
            )

            labels = tgt_inputs.input_ids.masked_fill(
                tgt_inputs.input_ids == probe_tokenizer.pad_token_id, -100
            )

            attention_mask_cuda = src_inputs.attention_mask.to("cuda")
            hidden_state_cuda = hidden_states.to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = probe(
                    # Encoder attention mask
                    attention_mask=attention_mask_cuda,
                    # Shifted internally to create decoder input
                    labels=labels,
                    encoder_outputs=(hidden_state_cuda,),
                )

            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_iterator.set_postfix({"loss": round(loss.item(), 2)})
            step += 1
            if step % log_interval == 0:
                wandb.log({"loss": loss.item()}, step=step)

            del (
                loss,
                output,
                hidden_state_cuda,
                attention_mask_cuda,
                tgt_inputs,
                labels,
            )
        if early_stopping:
            break

    artifact = wandb.Artifact(
        "best_probe",
        type="model",
        description="Best probe model",
        metadata=dict(
            model_name=probe_model_name,
            num_decoder_layers=probe_num_decoder_layers,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
        ),
    )
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--use_last_layer", action="store_true")
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    main(parser)
