import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PreferenceRewardModel(nn.Module):
    """Ensemble neural network reward model for robotic navigation preferences.
    
    Estimates the scalar utility r_psi(s, a) of trajectory transitions under the
    Bradley-Terry preference model.
    """
    def __init__(self, state_dim: int = 41, action_dim: int = 5, hidden_dim: int = 256, num_ensemble: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        
        # Multi-head ensemble to quantify epistemic reward uncertainty
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_ensemble)
        ])
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute mean predicted reward across the ensemble.
        
        Args:
            state: Tensor of shape (batch, state_dim)
            action: One-hot or continuous action tensor (batch, action_dim)
            
        Returns:
            Mean reward tensor of shape (batch, 1)
        """
        if action.dtype == torch.long:
            action = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state, action], dim=-1)
        rewards = torch.stack([model(x) for model in self.models], dim=0)
        return torch.mean(rewards, dim=0)

    def compute_ensemble_rewards(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and standard deviation across ensemble heads."""
        if action.dtype == torch.long:
            action = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state, action], dim=-1)
        rewards = torch.stack([model(x) for model in self.models], dim=0)
        return torch.mean(rewards, dim=0), torch.std(rewards, dim=0)
