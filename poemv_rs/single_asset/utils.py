from __future__ import annotations
import numpy as np

def set_seed(seed: int = 0):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def annualize_mean(daily_mean: float, days: int = 252) -> float:
    return daily_mean * days

def annualize_std(daily_std: float, days: int = 252) -> float:
    return daily_std * np.sqrt(days)

def safe_clip_p(p: float, eps: float = 1e-12) -> float:
    return float(np.clip(p, eps, 1.0 - eps))
