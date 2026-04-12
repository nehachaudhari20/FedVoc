"""
oov_eval.py — OOV analysis utilities.

Changes from original:
    - compute_oov_tokens() kept for backwards compatibility
    - NEW: oov_rate() measures rate on actual test sentences (the meaningful metric)
      Original measured vocab type overlap which gave ~2600-2700 for all clients
      (nearly identical and therefore useless as a differentiator).
"""


def compute_oov_tokens(client_tokenizer, other_tokenizers):
    """
    Original metric: count vocab types in client that don't appear in other vocabs.
    Kept for backwards compatibility but oov_rate() is more meaningful.
    """
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
    Measure the fraction of tokens in test_texts that fall outside the
    client's vocabulary. This is the meaningful OOV metric — it tells you
    how much of the actual test data the tokenizer can represent.

    Args:
        tokenizer:  a tokenizers.Tokenizer instance
        test_texts: list of strings to evaluate on
        max_texts:  cap for speed

    Returns:
        float in [0, 1] — lower is better (fewer unknown tokens)
    """
    unk_id = tokenizer.token_to_id("[UNK]")
    vocab = set(tokenizer.get_vocab().keys())

    total_tokens = 0
    oov_tokens = 0

    for text in test_texts[:max_texts]:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens

        for tok in tokens:
            total_tokens += 1
            if tok not in vocab or tokenizer.token_to_id(tok) == unk_id:
                oov_tokens += 1

    return oov_tokens / max(total_tokens, 1)


def oov_rate_summary(clients, max_texts=500):
    """
    Print OOV rate for each client on their own test set.
    Domain-specific tokenizers should have lower OOV rates on their domain
    than a global tokenizer would — this is the key FedVoc claim to verify.
    """
    domain_names = ["Shakespeare", "News", "Medical"]
    print(f"\n{'Domain':<14} {'OOV rate':>10}")
    print("-" * 26)
    for i, (client, name) in enumerate(zip(clients, domain_names)):
        rate = oov_rate(client.tokenizer, client.test_texts, max_texts)
        print(f"{name:<14} {rate:>10.4f}")
