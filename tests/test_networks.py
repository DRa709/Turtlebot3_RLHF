import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from turtlebot3_drl_nav.algorithms.discrete_sac.preference_learning.reward_model import PreferenceRewardModel
from turtlebot3_drl_nav.algorithms.discrete_sac.preference_learning.preference_loss import BradleyTerryLoss

def test_preference_reward_model_forward():
    batch_size = 8
    state_dim = 41
    action_dim = 5
    
    model = PreferenceRewardModel(state_dim=state_dim, action_dim=action_dim, hidden_dim=64, num_ensemble=3)
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    
    # Forward pass
    rewards = model(states, actions)
    assert rewards.shape == (batch_size, 1)
    
    # Ensemble rewards with uncertainty
    mean_r, std_r = model.compute_ensemble_rewards(states, actions)
    assert mean_r.shape == (batch_size, 1)
    assert std_r.shape == (batch_size, 1)
    assert torch.all(std_r >= 0.0)

def test_bradley_terry_loss():
    loss_fn = BradleyTerryLoss(label_smoothing=0.0)
    
    # If r1 > r2 and label is 1.0 (segment 1 preferred), loss should be small
    r1 = torch.tensor([5.0, 10.0])
    r2 = torch.tensor([1.0, 2.0])
    labels = torch.tensor([1.0, 1.0])
    
    loss = loss_fn(r1, r2, labels)
    assert loss.item() < 0.1
    
    # If r1 < r2 but label is 1.0, loss should be large
    r1_bad = torch.tensor([-5.0])
    r2_bad = torch.tensor([5.0])
    label_bad = torch.tensor([1.0])
    bad_loss = loss_fn(r1_bad, r2_bad, label_bad)
    assert bad_loss.item() > 5.0

def test_discrete_action_dimensions():
    state_dim = 41
    action_dim = 5
    
    # Synthetic discrete categorical policy head
    policy_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim)
    )
    
    obs = torch.randn(4, state_dim)
    logits = policy_net(obs)
    probs = F.softmax(logits, dim=-1)
    
    assert probs.shape == (4, action_dim)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    
    # Exact discrete categorical entropy
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    assert entropy.shape == (4,)
    assert torch.all(entropy >= 0.0)
