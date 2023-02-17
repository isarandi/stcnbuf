import torch


# Soft aggregation from STM
def aggregate(object_probs):
    bg_prob = torch.prod(1 - object_probs, dim=0, keepdim=True)
    new_prob = torch.cat([bg_prob, object_probs], 0).clamp(1e-7, 1 - 1e-7)
    odds = new_prob / (1 - new_prob)
    return odds / odds.sum(dim=0, keepdim=True)
