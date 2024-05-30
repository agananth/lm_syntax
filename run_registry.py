RUNS = {
    "EleutherAI/pythia-70m": "ananthag/Head Word Final 2/s8qh8azh",
    "EleutherAI/pythia-160m": "ananthag/Head Word Final 2/78zt5b5f",
    "EleutherAI/pythia-410m": "ananthag/Head Word Final 2/dkte7uzy",
    "EleutherAI/pythia-1.4b": "ananthag/Head Word Final 2/52xf5zjn",
    "EleutherAI/pythia-2.8b": "ananthag/Head Word Final 2/91dqlyy8",
    "tiiuae/falcon-7b": "ananthag/Head Word Final 2/jcptmzek",
    "gpt2-large": "ananthag/Head Word Final 2/r0u0bjcz",
    "gpt2-medium": "ananthag/Head Word Final 2/47z3cdgr",
    "gpt2": "ananthag/Head Word Final 2/2lhuc940",
    "EleutherAI/gpt-j-6b": "ananthag/Head Word Final 2/81vkf85b",
    "meta-llama/Llama-2-7b-hf": "ananthag/Head Word Final 2/p16wh1qg",
    "gpt2-xl": "ananthag/Head Word Final 2/xozj5ub1",
    "mistralai/Mistral-7B-v0.1": "ananthag/Head Word Final 2/4hcvqiwu",
    "google/gemma-2b": "ananthag/Head Word Final 2/y2opgjol",
    "microsoft/phi-2": "ananthag/Head Word Final 2/v7711eze",
    "meta-llama/Meta-Llama-3-8B": "ananthag/Head Word Final 2/6xavrhta",
    "google/electra-small-generator": "ananthag/Head Word Final 2/ie6tblg5",
    "google/electra-large-generator": "ananthag/Head Word Final 2/ppfaj83r",
    "google/electra-base-generator": "ananthag/Head Word Final 2/e0ynceav",
    "google-t5/t5-small": "ananthag/Head Word Final 2/6r587xkm",
    "google-t5/t5-base": "ananthag/Head Word Final 2/tpdl8ahl",
    "google-t5/t5-large": "ananthag/Head Word Final 2/rct7bx66",
    "google-t5/t5-3b": "ananthag/Head Word Final 2/aq5t48nq",
    "FacebookAI/roberta-large": "ananthag/Head Word Final 2/9v19tcnu",
    "FacebookAI/roberta-base": "ananthag/Head Word Final 2/5a6qww73",
    "EleutherAI/pythia-6.9b": "ananthag/Head Word Final 2/bu782p7y",
    "allenai/OLMo-1B-hf": "ananthag/Head Word Final 2/lorei94l",
    "allenai/OLMo-1.7-7B-hf": "ananthag/Head Word Final 2/mz4ohbnp",
}

RANDOM_WEIGHT_RUNS = {
    "gpt2": "ananthag/Head Word Final w Base Weights/gzzfn6k7",
    "gpt2-medium": "ananthag/Head Word Final w Base Weights/k1s9rzne",
    "gpt2-large": "ananthag/Head Word Final w Base Weights/423dyca8",
    "gpt2-xl": "ananthag/Head Word Final w Base Weights/kfidm7dh",
    "microsoft/phi-2": "ananthag/Head Word Final w Base Weights/5mmj3w3u",
    "meta-llama/Llama-2-7b-hf": "ananthag/Head Word Final w Base Weights/fyyeocd4",
}

sg_mapping = {
    "fgd_hierarchy.json": "Long-Distance Dependencies",
    "subordination_src-src.json": "Gross Syntactic State",
    "reflexive_src_fem.json": "Licensing",
    "npz_obj_mod.json": "Garden-Path Effects",
    "npi_src_any.json": "Licensing",
    "reflexive_prep_fem.json": "Licensing",
    "fgd-embed3.json": "Long-Distance Dependencies",
    "subordination.json": "Gross Syntactic State",
    "reflexive_prep_masc.json": "Licensing",
    "cleft_modifier.json": "Long-Distance Dependencies",
    "fgd_subject.json": "Long-Distance Dependencies",
    "mvrr.json": "Garden-Path Effects",
    "npi_src_ever.json": "Licensing",
    "npz_ambig.json": "Garden-Path Effects",
    "cleft.json": "Long-Distance Dependencies",
    "npi_orc_ever.json": "Licensing",
    "mvrr_mod.json": "Garden-Path Effects",
    "center_embed.json": "Center Embedding",
    "fgd-embed4.json": "Long-Distance Dependencies",
    "reflexive_src_masc.json": "Licensing",
    "subordination_pp-pp.json": "Gross Syntactic State",
    "subordination_orc-orc.json": "Gross Syntactic State",
    "number_orc.json": "Agreement",
    "npz_obj.json": "Garden-Path Effects",
    "reflexive_orc_masc.json": "Licensing",
    "number_prep.json": "Agreement",
    "fgd_object.json": "Long-Distance Dependencies",
    "npz_ambig_mod.json": "Garden-Path Effects",
    "center_embed_mod.json": "Center Embedding",
    "number_src.json": "Agreement",
    "reflexive_orc_fem.json": "Licensing",
    "fgd_pp.json": "Long-Distance Dependencies",
    "npi_orc_any.json": "Licensing",
}

sg_num_items = {
    "fgd_hierarchy.json": 24,
    "subordination_src-src.json": 23,
    "reflexive_src_fem.json": 19,
    "npz_obj_mod.json": 24,
    "npi_src_any.json": 38,
    "reflexive_prep_fem.json": 19,
    "fgd-embed3.json": 21,
    "subordination.json": 23,
    "reflexive_prep_masc.json": 19,
    "cleft_modifier.json": 40,
    "fgd_subject.json": 24,
    "mvrr.json": 28,
    "npi_src_ever.json": 38,
    "npz_ambig.json": 24,
    "cleft.json": 40,
    "npi_orc_ever.json": 38,
    "mvrr_mod.json": 28,
    "nn-nv-rpl.json": 1,
    "center_embed.json": 28,
    "fgd-embed4.json": 21,
    "reflexive_src_masc.json": 19,
    "subordination_pp-pp.json": 23,
    "subordination_orc-orc.json": 23,
    "number_orc.json": 19,
    "npz_obj.json": 24,
    "reflexive_orc_masc.json": 19,
    "number_prep.json": 19,
    "fgd_object.json": 24,
    "npz_ambig_mod.json": 24,
    "center_embed_mod.json": 28,
    "number_src.json": 19,
    "reflexive_orc_fem.json": 19,
    "fgd_pp.json": 24,
    "npi_orc_any.json": 38,
}
