from typing import Any

import numpy as np
from pydantic import BaseModel
import torch


class RemotePolicyConfig(BaseModel):
    """Connection settings for the remote policy websocket server."""

    host: str = "localhost"
    port: int = 8000


class AutoConfig(BaseModel):
    chunk_size_predict: int = 0
    state_dim: int = -1
    camera_names: list[str] = []
    observation: dict[str, Any] = {"qpos": None, "images": {}}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def interpolate_action(step_length, prev_action, cur_action):
    """Generate evenly spaced intermediate waypoints between two joint configurations."""
    steps = np.asarray(step_length, dtype=np.float32)
    if steps.size != prev_action.size:
        repeats = int(np.ceil(prev_action.size / steps.size))
        steps = np.tile(steps, repeats)[: prev_action.size]
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]
