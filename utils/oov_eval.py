def compute_oov_tokens(client_tokenizer, other_tokenizers):
    client_vocab = set(client_tokenizer.get_vocab().keys())

    other_vocab = set()
    for tok in other_tokenizers:
        other_vocab |= set(tok.get_vocab().keys())

    oov = client_vocab - other_vocab
    return oov
