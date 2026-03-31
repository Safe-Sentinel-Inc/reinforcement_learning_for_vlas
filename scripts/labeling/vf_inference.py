"""Run a trained value function over dataset frames and optionally compute action quality labels."""

import logging
import os
import pathlib

import numpy as np
import tqdm

from .dataset_utils import (
    read_episode_lengths,
    read_fold_assignments,
)
from .advantage_labeling import merge_and_label

logger = logging.getLogger(__name__)


def load_value_function(vf_config_name: str, vf_checkpoint_dir: str):
    """Restore a value function model and its data pipeline from a saved checkpoint."""
    import jax.numpy as jnp
    import openpi.models.model as _model
    import openpi.training.config as _config

    train_config = _config.get_config(vf_config_name)
    model_config = train_config.model
    params = _model.restore_params(
        pathlib.Path(vf_checkpoint_dir) / "params", dtype=jnp.bfloat16
    )
    vf_model = model_config.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    logger.info("Value function loaded from %s", vf_checkpoint_dir)
    return vf_model, data_config, model_config


class _VFInferDataset:
    """Wraps a LeRobot dataset for batched value function inference via PyTorch DataLoader."""

    def __init__(self, lerobot_ds, transform_fn, tasks_mapping, frame_order):
        self.ds = lerobot_ds
        self.transform_fn = transform_fn
        self.tasks_mapping = tasks_mapping
        self.frame_order = frame_order  # ordered pairs of (global_frame_index, episode_index)

    def __len__(self):
        return len(self.frame_order)

    def __getitem__(self, idx):
        dataset_fi, ep_idx = self.frame_order[idx]
        sample = self.ds[dataset_fi]
        sample_dict = {}
        for key in sample:
            val = sample[key]
            sample_dict[key] = val.numpy() if hasattr(val, "numpy") else val
        if "prompt" not in sample_dict and "task_index" in sample_dict:
            task_idx = int(sample_dict["task_index"])
            sample_dict["prompt"] = self.tasks_mapping.get(task_idx, "")
        result = self.transform_fn(sample_dict)
        result["_ep_idx"] = np.int64(ep_idx)
        return result


def _np_collate(batch):
    """Stack a list of sample dicts into a single dict of batched numpy arrays."""
    elem = batch[0]
    if isinstance(elem, dict):
        return {k: _np_collate([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, (np.ndarray, np.generic)):
        return np.stack([np.asarray(x) for x in batch])
    else:
        return batch


def _dl_worker_init(worker_id: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _run_vf_inference(
    repo_id: str,
    vf_config_name: str,
    vf_checkpoint_dir: str,
    batch_size: int,
    return_min: float,
    return_max: float,
    target_episodes: list[int],
    values_dir: str,
):
    """Evaluate the value function on the given episodes and write per-episode .npy files."""
    import multiprocessing
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import torch.utils.data
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.transforms as _transforms
    from openpi.shared import array_typing as at
    import openpi.models.model as _model

    dataset = lerobot_dataset.LeRobotDataset(repo_id)

    episode_lengths = read_episode_lengths(repo_id)
    target_frames = sum(episode_lengths[ep] for ep in target_episodes)
    logger.info(
        "VF inference: %d episodes, %d frames (total dataset: %d eps, %d frames)",
        len(target_episodes), target_frames,
        len(episode_lengths), sum(episode_lengths.values()),
    )

    vf_model, data_config, model_config = load_value_function(
        vf_config_name, vf_checkpoint_dir
    )

    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        ),
        *data_config.model_transforms.inputs,
    ]
    transform_fn = _transforms.compose(transforms_list)

    tasks_mapping = dataset.meta.tasks
    logger.info("Task mapping: %s", tasks_mapping)

    rng = jax.random.key(0)

    @nnx.jit
    def infer_batch(model, obs):
        return model.infer_value(rng, obs)

    logger.info("JIT warmup: compiling VF inference graph...")
    import time as _time
    _t0 = _time.time()
    fake_obs = model_config.fake_obs(batch_size=batch_size)
    _ = jax.device_get(infer_batch(vf_model, fake_obs))
    logger.info("JIT warmup done in %.1fs", _time.time() - _t0)

    # Collect the global frame indices for each target episode in order
    ep_indices = np.array(dataset.hf_dataset["episode_index"])
    frame_order = []
    for ep_idx in target_episodes:
        ep_mask = ep_indices == ep_idx
        for fi in np.where(ep_mask)[0]:
            frame_order.append((int(fi), int(ep_idx)))

    if not frame_order:
        logger.info("No frames to infer, skipping.")
        return

    num_dl_workers = min(16, len(os.sched_getaffinity(0)))
    logger.info("Using PyTorch DataLoader with %d workers, prefetch_factor=4", num_dl_workers)

    infer_ds = _VFInferDataset(dataset, transform_fn, tasks_mapping, frame_order)
    loader = torch.utils.data.DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dl_workers,
        multiprocessing_context=multiprocessing.get_context("spawn"),
        collate_fn=_np_collate,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=_dl_worker_init,
        drop_last=False,
    )

    def _batch_to_obs(batched: dict):
        """Transfer a batched numpy dict to JAX arrays and wrap it as an Observation."""
        converted = {}
        for k, v in batched.items():
            if isinstance(v, dict):
                converted[k] = {sk: jnp.asarray(sv) for sk, sv in v.items()}
            elif isinstance(v, np.ndarray):
                converted[k] = jnp.asarray(v)
            else:
                converted[k] = v
        with at.disable_typechecking():
            return _model.Observation.from_dict(converted)

    ep_values_map: dict[int, list[np.ndarray]] = {}
    values_path = pathlib.Path(values_dir)
    values_path.mkdir(parents=True, exist_ok=True)
    pbar = tqdm.tqdm(total=target_frames, desc="VF inference", unit="frame")
    saved_eps = 0

    for batch in loader:
        ep_idxs = batch.pop("_ep_idx")  # [B]
        cur_bs = len(ep_idxs)

        # Zero-pad undersized batches to keep a fixed shape for the JIT-compiled function
        if cur_bs < batch_size:
            def _pad(arr):
                if isinstance(arr, np.ndarray):
                    pad_width = [(0, batch_size - cur_bs)] + [(0, 0)] * (arr.ndim - 1)
                    return np.pad(arr, pad_width)
                return arr

            def _pad_tree(tree):
                if isinstance(tree, dict):
                    return {k: _pad_tree(v) for k, v in tree.items()}
                return _pad(tree)

            batch = _pad_tree(batch)

        obs = _batch_to_obs(batch)
        values = np.array(jax.device_get(infer_batch(vf_model, obs)))[:cur_bs]

        # Accumulate predicted values into per-episode lists
        for j in range(cur_bs):
            ep = int(ep_idxs[j])
            if ep not in ep_values_map:
                ep_values_map[ep] = []
            ep_values_map[ep].append(values[j])
        pbar.update(cur_bs)

        # Flush completed episodes to disk to free memory
        for ep in list(ep_values_map.keys()):
            if len(ep_values_map[ep]) == episode_lengths[ep]:
                ep_vals = np.array(ep_values_map[ep], dtype=np.float32)
                ep_vals = np.clip(ep_vals, return_min, return_max)
                np.save(values_path / f"ep_{ep:06d}.npy", ep_vals)
                saved_eps += 1
                pbar.set_postfix(ep=ep, vmin=f"{ep_vals.min():.4f}", vmax=f"{ep_vals.max():.4f}")
                del ep_values_map[ep]

    # Save any episodes still in memory after the final batch
    for ep, vals_list in ep_values_map.items():
        ep_vals = np.array(vals_list, dtype=np.float32)
        ep_vals = np.clip(ep_vals, return_min, return_max)
        np.save(values_path / f"ep_{ep:06d}.npy", ep_vals)
        saved_eps += 1

    pbar.close()
    logger.info("VF inference done. Saved %d episode values to %s", saved_eps, values_dir)


def infer_values_for_dataset(
    repo_id: str,
    vf_config_name: str,
    vf_checkpoint_dir: str,
    positive_fraction: float = 0.3,
    gamma: float = 0.99,
    return_min: float = 0.0,
    return_max: float = 1.0,
    output_dir: str | None = None,
    batch_size: int = 32,
    infer_fold: int | None = None,
    values_dir: str | None = None,
):
    """Predict values for a dataset and, unless doing fold-based sharding, produce final labels.

    If infer_fold is set, only that fold's episodes are processed and the raw
    values are saved to values_dir for later merging via the vf_merge subcommand.
    Otherwise, all episodes are inferred and advantage-based labels are computed
    in a single pass.
    """
    if values_dir is None:
        values_dir = f"/tmp/vf_values_{repo_id}"

    episode_lengths = read_episode_lengths(repo_id)

    if infer_fold is not None:
        # Fold-based sharding: restrict inference to episodes assigned to this fold
        fold_map = read_fold_assignments(repo_id)
        target_episodes = sorted(ep for ep, fold in fold_map.items() if fold == infer_fold)
        logger.info("Fold %d: %d episodes to infer", infer_fold, len(target_episodes))
        _run_vf_inference(
            repo_id=repo_id,
            vf_config_name=vf_config_name,
            vf_checkpoint_dir=vf_checkpoint_dir,
            batch_size=batch_size,
            return_min=return_min,
            return_max=return_max,
            target_episodes=target_episodes,
            values_dir=values_dir,
        )
        return

    # Full-dataset mode: infer all episodes then immediately compute labels
    all_episodes = sorted(episode_lengths.keys())
    _run_vf_inference(
        repo_id=repo_id,
        vf_config_name=vf_config_name,
        vf_checkpoint_dir=vf_checkpoint_dir,
        batch_size=batch_size,
        return_min=return_min,
        return_max=return_max,
        target_episodes=all_episodes,
        values_dir=values_dir,
    )
    merge_and_label(
        repo_id=repo_id,
        values_dir=values_dir,
        positive_fraction=positive_fraction,
        gamma=gamma,
        output_dir=output_dir,
    )


def _run_vf_label(args):
    """Entry point for the vf_label CLI subcommand."""
    infer_values_for_dataset(
        repo_id=args.repo_id,
        vf_config_name=args.vf_config,
        vf_checkpoint_dir=args.vf_checkpoint_dir,
        positive_fraction=args.positive_fraction,
        gamma=args.gamma,
        return_min=args.return_min,
        return_max=args.return_max,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        infer_fold=args.infer_fold,
        values_dir=args.values_dir,
    )
