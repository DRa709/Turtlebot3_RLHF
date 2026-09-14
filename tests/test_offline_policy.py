import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import torch
from turtlebot3_drl_nav.discretesac import DiscreteSACConfig, CategoricalActor
from turtlebot3_drl_nav.state import OBSERVATION_DIM, ACTION_NAMES


def test_discretesac_config_validation():
    cfg = DiscreteSACConfig()
    cfg.validate()
    assert cfg.gamma == 0.99
    assert cfg.alpha == 0.2
    assert cfg.hidden_size == 256
    assert cfg.batch_size == 64
    assert cfg.warmup_steps == 5000


def test_categorical_actor_dimensions_and_distribution():
    torch.set_num_threads(1)
    cfg = DiscreteSACConfig()
    actor = CategoricalActor(OBSERVATION_DIM, len(ACTION_NAMES), cfg.hidden_size)
    actor.eval()

    obs = np.random.uniform(0.0, 1.0, size=(OBSERVATION_DIM,)).astype(np.float32)
    tensor = torch.from_numpy(obs).unsqueeze(0)

    with torch.no_grad():
        probs, log_probs = actor.distribution(tensor)

    assert probs.shape == (1, 5)
    assert log_probs.shape == (1, 5)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert (probs >= 0.0).all()
