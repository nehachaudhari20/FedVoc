def compute_oov_tokens(client_tokenizer, other_tokenizers):

    def normalize(vocab):
        return set(t.lower() for t in vocab if t not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]"])

    client_vocab = normalize(client_tokenizer.get_vocab().keys())

    other_vocab = set()
    for tok in other_tokenizers:
        other_vocab |= normalize(tok.get_vocab().keys())

    oov = client_vocab - other_vocab

    return oov
