import torch
import torch.nn.functional as F

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # PyTorch's F.cross_entropy expects logits and not probabilities
    # It also combines log_softmax and nll_loss in one function for efficiency
    loss = F.cross_entropy(logits, targets)
    return loss
