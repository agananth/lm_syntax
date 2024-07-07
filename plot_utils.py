import wandb
import dataclasses
import utils
import run_registry
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np

api = wandb.Api()


PLOT_NAMES = {
    # "EleutherAI/pythia-70m": "Pythia 70M",
    # "EleutherAI/pythia-160m": "Pythia 160M",
    "EleutherAI/pythia-410m": "Pythia 410M",
    "EleutherAI/pythia-1.4b": "Pythia 1.4B",
    "EleutherAI/pythia-2.8b": "Pythia 2.8B",
    "EleutherAI/pythia-6.9b": "Pythia 6.9B",
    "tiiuae/falcon-7b": "Falcon 7B",
    "gpt2-large": "GPT-2 774M",
    "gpt2-medium": "GPT-2 345M",
    "gpt2": "GPT-2 117M",
    "EleutherAI/gpt-j-6b": "GPT-J 6B",
    "meta-llama/Llama-2-7b-hf": "Llama 2 7B",
    "meta-llama/Meta-Llama-3-8B": "Llama 3 8B",
    "gpt2-xl": "GPT-2 1.6B",
    "mistralai/Mistral-7B-v0.1": "Mistral 7B",
    "google/gemma-2b": "Gemma 2B",
    "microsoft/phi-2": "Phi-2 2.7B",
    "google/electra-small-generator": "Electra 14M",
    "google/electra-large-generator": "Electra 34M",
    "google/electra-base-generator": "Electra 51M",
    "google-t5/t5-small": "T5 35M",
    "google-t5/t5-base": "T5 110M",
    "google-t5/t5-large": "T5 335M",
    "google-t5/t5-3b": "T5 3B",
    "FacebookAI/roberta-large": "RoBERTa 355M",
    "FacebookAI/roberta-base": "RoBERTa 125M",
    "allenai/OLMo-1B-hf": "OLMo 1B",
    "allenai/OLMo-1.7-7B-hf": "OLMo 7B",
}


COLOR_MAPPING = {
    "EleutherAI/pythia-410m": "red",
    "EleutherAI/pythia-1.4b": "blue",
    "EleutherAI/pythia-2.8b": "blue",
    "EleutherAI/pythia-6.9b": "green",
    "tiiuae/falcon-7b": "green",
    "gpt2-large": "red",
    "gpt2-medium": "red",
    "gpt2": "red",
    "EleutherAI/gpt-j-6b": "green",
    "meta-llama/Llama-2-7b-hf": "green",
    "meta-llama/Meta-Llama-3-8B": "green",
    "gpt2-xl": "blue",
    "mistralai/Mistral-7B-v0.1": "green",
    "google/gemma-2b": "blue",
    "microsoft/phi-2": "blue",
    "google/electra-small-generator": "red",
    "google/electra-large-generator": "red",
    "google/electra-base-generator": "red",
    "google-t5/t5-small": "red",
    "google-t5/t5-base": "red",
    "google-t5/t5-large": "red",
    "google-t5/t5-3b": "blue",
    "FacebookAI/roberta-large": "red",
    "FacebookAI/roberta-base": "red",
    "allenai/OLMo-1B-hf": "blue",
    "allenai/OLMo-1.7-7B-hf": "green",
}


def get_all_probe_results(registry: dict = run_registry.RUNS) -> dict[str, list[float]]:
    probe_results = {}
    for model_name, run_id in registry.items():
        run = api.run(run_id)
        num_layers = run.config["num_layers"]
        layer_accs = []
        for layer in range(num_layers):
            layer_accs.append(run.summary_metrics[f"layer_{layer}/test/acc"])
        probe_results[model_name] = layer_accs
    return probe_results


def get_best_probe_results() -> dict[str, float]:
    probe_results = {}
    all_probe_results = get_all_probe_results()
    for model_name, layer_accs in all_probe_results.items():
        probe_results[model_name] = max(layer_accs)
    return probe_results


def get_best_probe_layer_indices() -> dict[str, int]:
    probe_results_best_layer_idx = {}
    all_probe_results = get_all_probe_results()
    for model_name, layer_accs in all_probe_results.items():
        best_layer_idx = layer_accs.index(max(layer_accs))
        probe_results_best_layer_idx[model_name] = best_layer_idx
    return probe_results_best_layer_idx


def get_blimp_aggregate_results(model_name: str) -> dict[str, float]:
    results = {}
    with open(f'blimp_results/{model_name.replace("/", "_")}_aggregate.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["linguistics_term"]] = float(row["accuracy"])
    return results


ENCODER_DECODER = "decoder"
ENCODER = "encoder"
DECODER = "encoder_decoder"


def get_model_type(model_name):
    model_name = model_name.lower()
    if "t5" in model_name:
        return ENCODER_DECODER
    if "roberta" in model_name or "electra" in model_name:
        return ENCODER
    return DECODER


MARKER_MAPPING = {
    ENCODER_DECODER: "s",
    ENCODER: "^",
    DECODER: "o",
}


def plot(
    kwargs_list,
    annotations_list,
    x_label,
    y_label,
    title,
    save_path,
    specific_model_type=None,
):
    plt.figure(figsize=(8, 6))  # Set the figure size

    final_kwargs_list = []
    final_annotations_list = []
    for kwargs, annotation in zip(kwargs_list, annotations_list):
        model_name = annotation["text"]
        if not specific_model_type:
            final_kwargs_list.append(kwargs)
            final_annotations_list.append(annotation)
        elif get_model_type(model_name) == specific_model_type:
            final_kwargs_list.append(kwargs)
            final_annotations_list.append(annotation)

    for kwargs, annotation in zip(final_kwargs_list, final_annotations_list):
        plt.scatter(**kwargs)
        plt.annotate(**annotation)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="< 1B",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="1B - 6B",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="> 6B",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="Decoder",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="Encoder",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="T5",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left")

    plt.grid()
    plt.savefig(save_path)
    plt.show()


def spearman(xs, ys, model_names, specific_model_type=None):
    final_xs = []
    final_ys = []
    for x, y, model_name in zip(xs, ys, model_names):
        if not specific_model_type:
            final_xs.append(x)
            final_ys.append(y)
        elif get_model_type(model_name) == specific_model_type:
            final_xs.append(x)
            final_ys.append(y)

    spearman_coeff = stats.spearmanr(final_xs, final_ys).statistic

    dof = len(final_xs) - 2

    def statistic(x):  # explore all possible pairings by permuting `x`
        rs = stats.spearmanr(x, final_ys).statistic  # ignore pvalue
        transformed = rs * np.sqrt(dof / ((rs + 1.0) * (1.0 - rs)))
        return transformed

    p_value = stats.permutation_test(
        (final_xs,), statistic, permutation_type="pairings"
    )
    return spearman_coeff, p_value


if __name__ == "__main__":
    print(get_best_probe_layer_indices())
