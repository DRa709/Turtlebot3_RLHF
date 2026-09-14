import torch
import torch.nn as nn
import torch.nn.functional as F

class BradleyTerryLoss(nn.Module):
    """Negative log-likelihood of the Bradley-Terry preference model.
    
    P(sigma_1 > sigma_2) = exp(sum r(s, a)) / (exp(sum r(s, a)) + exp(sum r(s', a')))
    = sigmoid(sum r_1 - sum r_2)
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, r1_sum: torch.Tensor, r2_sum: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Args:
            r1_sum: Cumulative reward for segment 1, shape (batch,)
            r2_sum: Cumulative reward for segment 2, shape (batch,)
            labels: Binary preference target, 1.0 if sigma_1 preferred, 0.0 if sigma_2 preferred
        """
        logits = r1_sum - r2_sum
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, labels)
