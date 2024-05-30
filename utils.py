import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    GPT2Tokenizer,
    GemmaTokenizer,
    RobertaTokenizer,
    ElectraTokenizer,
    T5Tokenizer,
    AutoConfig,
    PreTrainedTokenizerFast,
    T5EncoderModel,
)


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast="pythia" in model_name or "olmo" in model_name.lower(),
        token="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",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(model_name: str):
    if "google-t5" in model_name:
        return T5EncoderModel.from_pretrained(model_name).cuda()
    return AutoModel.from_pretrained(
        model_name, token="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"
    ).cuda()


def get_model_base_weights(model_name: str):
    if "google-t5" in model_name:
        return T5EncoderModel.from_config(AutoConfig.from_pretrained(model_name)).cuda()
    return AutoModel.from_config(
        AutoConfig.from_pretrained(
            model_name, token="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"
        )
    ).cuda()


def get_num_layers(config):
    if hasattr(config, "n_layer"):
        return config.n_layer
    return config.num_hidden_layers


def get_idxs(phrase_tokens, sent_tokens, st):
    while st < len(sent_tokens):
        en = st + len(phrase_tokens)
        if sent_tokens[st:en] == phrase_tokens:
            return (st, en)
        st += 1
    ### should not get to this point!
    raise AssertionError(
        f"phrase_tokens {phrase_tokens} not found in sent_tokens {sent_tokens}"
    )


def is_pythia_tokenizer(tokenizer):
    return isinstance(tokenizer, GPTNeoXTokenizerFast)


def is_llama_tokenizer(tokenizer):
    return isinstance(tokenizer, LlamaTokenizer)


def is_gemma_tokenizer(tokenizer):
    return isinstance(tokenizer, GemmaTokenizer)


def is_falcon_tokenizer(tokenizer):
    return "tiiuae/falcon-7b" == tokenizer.name_or_path


def is_phi_tokenizer(tokenizer):
    return "microsoft/phi-2" == tokenizer.name_or_path


def is_roberta_tokenizer(tokenizer):
    return isinstance(tokenizer, RobertaTokenizer)


def is_electra_tokenizer(tokenizer):
    return isinstance(tokenizer, ElectraTokenizer)


def is_t5_tokenizer(tokenizer):
    return isinstance(tokenizer, T5Tokenizer)


def is_llama3_tokenizer(tokenizer):
    return isinstance(tokenizer, PreTrainedTokenizerFast)


def get_tokenized_word(tokenizer, word: str, index: int):
    if (
        is_pythia_tokenizer(tokenizer)
        or is_falcon_tokenizer(tokenizer)
        or is_phi_tokenizer(tokenizer)
    ):
        if index:
            word = " " + word
        return tokenizer(word).input_ids
    elif is_llama_tokenizer(tokenizer):
        assert tokenizer.add_bos_token
        input_ids = tokenizer(word).input_ids
        assert input_ids.pop(0) == tokenizer.bos_token_id
        return input_ids
    elif is_llama3_tokenizer(tokenizer):
        if index:
            word = " " + word
        input_ids = tokenizer(word).input_ids
        assert input_ids.pop(0) == tokenizer.bos_token_id
        return input_ids
    elif is_gemma_tokenizer(tokenizer):
        assert tokenizer.add_bos_token
        if index:
            word = " " + word
        input_ids = tokenizer(word).input_ids
        assert input_ids.pop(0) == tokenizer.bos_token_id
        return input_ids
    elif is_roberta_tokenizer(tokenizer):
        if index:
            word = " " + word
        input_ids = tokenizer(word).input_ids
        assert input_ids.pop(0) == tokenizer.bos_token_id
        assert input_ids.pop() == tokenizer.eos_token_id
        return input_ids
    elif is_electra_tokenizer(tokenizer):
        if index:
            word = " " + word
        input_ids = tokenizer(word).input_ids
        assert input_ids.pop(0) == tokenizer.cls_token_id
        assert input_ids.pop() == tokenizer.sep_token_id
        return input_ids
    elif is_t5_tokenizer(tokenizer):
        if index:
            word = " " + word
        input_ids = tokenizer(word).input_ids
        assert input_ids.pop() == tokenizer.eos_token_id
        return input_ids
    assert isinstance(tokenizer, GPT2Tokenizer)
    return tokenizer(word, add_prefix_space=bool(index)).input_ids


def get_pre_tokenized_info(tokenizer, input_str: str):
    """
    e.g.
        input_str: The man is eating bananas
        [The, man, is, eating, bananas]
        [The, man, is, eat##, ##ing, bananas]
        [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6)]
    """
    sent_tokens = tokenizer(input_str).input_ids
    words = input_str.split(" ")
    idxs = []
    # go in order.
    st = 0
    for i, word in enumerate(words):
        # GPT2 tokenizer treats spaces like parts of the tokens
        word_tokenized = get_tokenized_word(tokenizer, word=word, index=i)
        st_curr, en_curr = get_idxs(word_tokenized, sent_tokens, st)
        idxs.append((st_curr, en_curr))
        st = en_curr
    return sent_tokens, idxs


def get_word_hidden_states(input_batch, tokenizer, model):
    token_input = tokenizer(input_batch, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        token_hidden_states_batch = model(
            **token_input, output_hidden_states=True
        ).hidden_states
    num_layers = len(token_hidden_states_batch) - 1
    # token_hidden_states_batch: (num_layers, batch_size, num_tokens, hidden_size)
    word_level_hs_batch = []
    # output shape is (batch_size, num_layers, num_words, hidden_size)
    for i, input_s in enumerate(input_batch):
        idxs = get_pre_tokenized_info(tokenizer, input_s)
        num_words = len(input_s.split(" "))
        assert num_words == len(idxs)
        word_hs_combined_from_tokens = []
        for layer in range(num_layers):
            word_level_hidden_states = torch.zeros(
                (num_words, model.config.hidden_size)
            )
            remove_token_embeddings = (
                token_hidden_states_batch[layer + 1][i]
                - token_hidden_states_batch[0][i]
            )
            for word_idx, (st, en) in enumerate(idxs):
                word_level_hidden_states[word_idx] = remove_token_embeddings[
                    st:en
                ].mean(dim=0)
            word_hs_combined_from_tokens.append(word_level_hidden_states)
        word_level_hs_batch.append(word_hs_combined_from_tokens)
    return word_level_hs_batch
