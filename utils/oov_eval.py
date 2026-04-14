def compute_oov_tokens(client_tokenizer, other_tokenizers):
    """Original metric — kept for backwards compatibility."""
    def normalize(vocab):
        return set(
            t.lower() for t in vocab
            if t not in ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
        )
    client_vocab = normalize(client_tokenizer.get_vocab().keys())
    other_vocab = set()
    for tok in other_tokenizers:
        other_vocab |= normalize(tok.get_vocab().keys())
    return client_vocab - other_vocab


def oov_rate(tokenizer, test_texts, max_texts=500):
    """
    Fraction of test tokens that are OOV.
    This is the meaningful metric — original only counted vocab types
    which gave ~2600-2700 for all clients (identical = useless).
    """
    unk_id = tokenizer.token_to_id("[UNK]")
    vocab = set(tokenizer.get_vocab().keys())
    total_tokens = 0
    oov_tokens = 0

    for text in test_texts[:max_texts]:
        for tok in tokenizer.encode(text).tokens:
            total_tokens += 1
            if tok not in vocab or tokenizer.token_to_id(tok) == unk_id:
                oov_tokens += 1

    return oov_tokens / max(total_tokens, 1)
