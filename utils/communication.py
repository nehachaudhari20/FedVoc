def count_parameters(state_dict):
    """Count total number of parameters in a state dict."""
    total = 0
    for v in state_dict.values():
        total += v.numel()
    return total
