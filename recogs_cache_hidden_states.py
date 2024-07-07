from torch.utils.data import Dataset, DataLoader
import argparse
import utils
import os
from tqdm import tqdm
import numpy as np
import torch
import wandb


class COGSDataset(Dataset):
    def __init__(self, split):
        self.srcs = []
        for l in open(f"ReCOGS/recogs_v2/{split}.tsv", "r").readlines():
            text, sparse, _ = l.split("\t")
            self.srcs.append(text)

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        return self.srcs[idx]


def main():
    args = parser.parse_args()
    model_name = args.model
    tokenizer = utils.get_tokenizer(model_name)
    if args.random_model:
        model = utils.get_model_base_weights(model_name)

        wandb.init(project="ReCOGS Random Models", name=model_name)
        path = f"{model_name.replace('/', '_')}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        wandb.save(path)
    else:
        model = utils.get_model(model_name)

    train_dataset = COGSDataset("train")
    val_dataset = COGSDataset("dev")

    folder_path = os.path.join(
        "/scr/biggest/ananthag",
        # "hidden_states",
        "recogs",
        "random" if args.random_model else "",
        model_name.replace("/", "_"),
    )
    if os.path.exists(folder_path):
        os.rmdir(folder_path)
    os.makedirs(folder_path)
    cache_files = (
        os.path.join(folder_path, "train.dat"),
        os.path.join(folder_path, "dev.dat"),
    )

    model.eval()

    for cache_file, dataset in zip(cache_files, (train_dataset, val_dataset)):
        open(cache_file, "w").close()
        offset = 0
        for item in tqdm(dataset):
            tokenized = tokenizer(item, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                last_hidden_states = model(
                    **tokenized, output_hidden_states=True
                ).last_hidden_state

            to_cache = last_hidden_states.squeeze(0).cpu()
            assert to_cache.shape == (
                tokenized.input_ids.shape[-1],
                model.config.hidden_size,
            ), (to_cache.shape, tokenized.input_ids.shape, model.config.hidden_size)
            cache = np.memmap(
                cache_file,
                mode="r+",
                shape=to_cache.shape,
                dtype=np.float32,
                offset=offset,
            )
            cache[:] = to_cache.numpy()
            offset += to_cache.element_size() * to_cache.nelement()

    if args.random_model:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--random_model", action="store_true")
    main()
