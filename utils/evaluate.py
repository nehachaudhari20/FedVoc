import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math


def evaluate_model(model, tokenizer, texts, device="cpu", max_len=32):

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_loss = 0
    steps = 0

    with torch.no_grad():

        for text in texts:

            ids = tokenizer.encode(text).ids[:max_len]

            if len(ids) < 2:
                continue

            input_tensor = torch.tensor([ids]).to(device)

            inputs = input_tensor[:, :-1]
            targets = input_tensor[:, 1:]

            attention_mask = (inputs != 0).long()

            logits = model(inputs, attention_mask)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity
