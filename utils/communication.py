def count_parameters(state_dict):
    total = 0
    for v in state_dict.values():
        total += v.numel()
    return total
