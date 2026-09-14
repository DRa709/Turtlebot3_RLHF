import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    """Dataset of pairwise trajectory comparisons (sigma_1, sigma_2, preference_label)."""
    def __init__(self, comparisons: List[Dict[str, Any]]):
        self.comparisons = comparisons

    def __len__(self) -> int:
        return len(self.comparisons)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.comparisons[idx]
        return {
            "seg1_states": torch.tensor(item["seg1_states"], dtype=torch.float32),
            "seg1_actions": torch.tensor(item["seg1_actions"], dtype=torch.long),
            "seg2_states": torch.tensor(item["seg2_states"], dtype=torch.float32),
            "seg2_actions": torch.tensor(item["seg2_actions"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float32)
        }

    @classmethod
    def load_from_json(cls, filepath: Path) -> "PreferenceDataset":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data if isinstance(data, list) else data.get("comparisons", []))
