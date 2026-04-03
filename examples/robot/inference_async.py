import collections
import logging
import threading
import time
from typing import Any

from airdc.common.systems.basis import System
from airdc.common.systems.basis import SystemMode
from airdc.utils import init_logging
from dagger_controller import DaggerConfig
from dagger_controller import DaggerController
from dagger_controller import DaggerMode
from inference_helpers import AutoConfig
from inference_helpers import RemotePolicyConfig
from inference_helpers import interpolate_action
from inference_helpers import set_seed
from keyboard_listener import KeyboardListener
import numpy as np
from pydantic import BaseModel
from robot_config import RobotConfig
import torch
import tyro

init_logging(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InferConfig(BaseModel):
    """Parameters for the asynchronous inference loop."""

    policy_config: RemotePolicyConfig
    max_steps: int = 500000
    step_rate: int = 20
    step_length: list[float] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05]

    reset_action: list[float] = [
        -0.001618136651813984,
        -1.0361113548278809,
        0.8421794176101685,
        1.6158959865570068,
        -0.6345375776290894,
        -1.6957406997680664,
        0.0,
        0.1323927342891693,
        -1.2208569049835205,
        1.0429750680923462,
        -2.0076663494110107,
        0.840582549571991,
        2.0390350818634033,
        0.0,
    ]
    interpolate: bool = False
    chunk_size_execute: int = 25
    # Negative value: request inference on every new observation.
    # Non-negative value: only request when the action queue shrinks below this count.
    inference_trigger_threshold: int = -1
    tcs_drop_max: int = 12
    tcs_min_overlap: int = 8
    initial_action_wait_s: float = 10.0
    debug: bool = False
    prompt: str = ""
    dagger: DaggerConfig = DaggerConfig(enable=False)
    robot_config: RobotConfig


auto_config = AutoConfig()

config = tyro.cli(InferConfig, config=[tyro.conf.ConsolidateSubcommandArgs])
robot_config = config.robot_config
if robot_config.robot_type == "play":
    from play_operator import Robot
else:
    raise ValueError("Unsupported robot type. Please use a valid config path for the robot.")


def clone_policy_observation(policy_observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "qpos": np.asarray(policy_observation["qpos"], dtype=np.float32).copy(),
        "images": {
            camera_name: np.asarray(image, dtype=np.uint8).copy()
            for camera_name, image in policy_observation["images"].items()
        },
    }


def update_observation(camera_names: list[str], operator: System) -> tuple[dict, dict[str, Any]]:
    """Refresh the shared observation cache and return both raw and policy-ready copies."""
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    for camera_name in camera_names:
        image_dict[camera_name] = np.asarray(obs[f"{camera_name}/color/image_raw"]["data"]).copy()
    auto_config.observation = {"qpos": np.asarray(qpos, dtype=np.float32), "images": image_dict}
    return obs, clone_policy_observation(auto_config.observation)


def build_policy_input(policy_observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "state": policy_observation["qpos"],
        "prompt": prompt,
        "advantage": True,
    } | policy_observation["images"]


def inference_once(policy, prompt: str, policy_observation: dict[str, Any] | None = None) -> np.ndarray:
    """Query the policy for an action chunk using the given observation."""
    observation = policy_observation or auto_config.observation
    obs = build_policy_input(observation, prompt)
    action_chunk = policy.infer(obs)["actions"] if policy is not None else np.zeros([64, 7], dtype=np.float32)
    auto_config.chunk_size_predict = action_chunk.shape[0]
    auto_config.state_dim = action_chunk.shape[1]
    return np.asarray(action_chunk, dtype=np.float32)


def smooth_action_chunks(
    old_actions: list[np.ndarray],
    new_chunk: np.ndarray,
    executed_since_request: int,
    drop_max: int,
    min_overlap: int,
) -> np.ndarray:
    """Blend a new action chunk with the remaining actions using temporal smoothing."""
    drop = min(max(executed_since_request, 0), drop_max)
    if drop >= len(new_chunk):
        return np.asarray(old_actions, dtype=np.float32) if old_actions else np.empty((0, 0), dtype=np.float32)

    new_remaining = np.asarray(new_chunk[drop:], dtype=np.float32)
    if not old_actions:
        return new_remaining

    old_buffer = np.asarray(old_actions, dtype=np.float32)
    old_for_overlap = old_buffer
    if len(old_for_overlap) < min_overlap:
        pad_count = min_overlap - len(old_for_overlap)
        pad = np.repeat(old_for_overlap[-1][np.newaxis, :], pad_count, axis=0)
        old_for_overlap = np.concatenate([old_for_overlap, pad], axis=0)

    overlap = min(len(old_for_overlap), len(new_remaining))
    if overlap <= 0:
        return new_remaining

    if overlap == 1:
        weights = np.array([1.0], dtype=np.float32)
    else:
        weights = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    blended = weights[:, np.newaxis] * old_for_overlap[:overlap] + (1.0 - weights)[:, np.newaxis] * new_remaining[:overlap]
    suffix = new_remaining[overlap:]
    if len(suffix) == 0:
        return blended
    return np.concatenate([blended, suffix], axis=0)


class AsyncChunkController:
    """Background thread that queries the policy and manages a smoothed action queue."""

    def __init__(self, policy, prompt: str, config: InferConfig, dagger_ctrl: DaggerController | None = None):
        self._policy = policy
        self._prompt = prompt
        self._config = config
        self._dagger_ctrl = dagger_ctrl

        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._latest_observation: dict[str, Any] | None = None
        self._latest_observation_version = 0
        self._executed_steps = 0

        self._lock = threading.Lock()
        self._observation_event = threading.Event()
        self._actions_ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True, name="async-policy-inference")

    def start(self):
        self._thread.start()

    def shutdown(self):
        self._stop_event.set()
        self._observation_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def update_observation(self, policy_observation: dict[str, Any]):
        with self._lock:
            self._latest_observation = clone_policy_observation(policy_observation)
            self._latest_observation_version += 1
        self._observation_event.set()

    def wait_for_actions(self, timeout: float) -> bool:
        return self._actions_ready.wait(timeout=timeout)

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            if not self._buffer:
                self._actions_ready.clear()
                return None
            action = self._buffer.popleft()
            if not self._buffer:
                self._actions_ready.clear()
            return np.asarray(action, dtype=np.float32).copy()

    def mark_step_executed(self):
        with self._lock:
            self._executed_steps += 1

    def clear_actions(self):
        with self._lock:
            self._buffer.clear()
            self._actions_ready.clear()

    def _should_pause(self) -> bool:
        return self._dagger_ctrl is not None and self._dagger_ctrl.mode != DaggerMode.INFERENCE

    def _should_request(self, obs_version: int, last_requested_version: int, remaining_actions: int) -> bool:
        if obs_version <= last_requested_version:
            return False
        if self._should_pause():
            return False
        if self._config.inference_trigger_threshold < 0:
            return True
        return remaining_actions <= self._config.inference_trigger_threshold

    def _inference_loop(self):
        last_requested_version = 0
        while not self._stop_event.is_set():
            if not self._observation_event.wait(timeout=0.05):
                continue
            self._observation_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                policy_observation = (
                    None if self._latest_observation is None else clone_policy_observation(self._latest_observation)
                )
                observation_version = self._latest_observation_version
                request_step = self._executed_steps
                remaining_actions = len(self._buffer)

            if policy_observation is None:
                continue
            if not self._should_request(observation_version, last_requested_version, remaining_actions):
                continue

            last_requested_version = observation_version
            infer_start = time.monotonic()
            action_chunk = inference_once(self._policy, self._prompt, policy_observation)
            infer_dt = time.monotonic() - infer_start

            if self._should_pause():
                continue

            with self._lock:
                latency_steps = self._executed_steps - request_step
                smoothed_actions = smooth_action_chunks(
                    list(self._buffer),
                    action_chunk,
                    latency_steps,
                    self._config.tcs_drop_max,
                    self._config.tcs_min_overlap,
                )
                self._buffer = collections.deque(np.asarray(action, dtype=np.float32).copy() for action in smoothed_actions)
                if self._buffer:
                    self._actions_ready.set()
                else:
                    self._actions_ready.clear()
                remaining_after_fuse = len(self._buffer)

            logger.info(
                "Async inference done: %.3fs, latency_steps=%d, chunk=%d, queue=%d",
                infer_dt,
                latency_steps,
                len(action_chunk),
                remaining_after_fuse,
            )


def model_inference(config: InferConfig, operator: System):
    auto_config.camera_names = operator.config.camera_names
    assert config.prompt, "Prompt must be provided for policy inference."

    from openpi_client import websocket_client_policy

    policy_config = config.policy_config
    logger.info("Connecting to policy server at %s:%s", policy_config.host, policy_config.port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=policy_config.host, port=policy_config.port)

    dagger_ctrl = None
    if config.dagger.enable:
        dagger_ctrl = DaggerController(config.dagger)
        operator.init_leaders()
        dagger_ctrl.start_keyboard_listener()
        logger.info(
            "[DAgger] DAgger mode ENABLED. Press '%s' to intervene, '%s' to resume, 'q' to quit.",
            config.dagger.key_enter_dagger,
            config.dagger.key_resume_inference,
        )

    keyboard_listener = None
    if not dagger_ctrl:
        keyboard_listener = KeyboardListener()
        keyboard_listener.start()

    try:
        while True:
            if dagger_ctrl and dagger_ctrl.should_quit:
                logger.info("DAgger quit requested.")
                break
            if keyboard_listener and keyboard_listener.check_quit():
                logger.info("Quitting...")
                break

            operator.switch_mode(SystemMode.RESETTING)
            operator.send_action(config.reset_action)

            if keyboard_listener:
                logger.info("Press 'Enter' to start episode...")
                while not keyboard_listener.check_start():
                    if keyboard_listener.check_quit():
                        logger.info("Quitting...")
                        break
                    time.sleep(0.1)
                if keyboard_listener.check_quit():
                    break
            else:
                if input("Press 'Enter' to continue or 'q' and 'Enter' to quit...") in {"q", "Q", "z", "Z"}:
                    logger.info("Quitting...")
                    break

            operator.switch_mode(SystemMode.SAMPLING)
            if dagger_ctrl:
                dagger_ctrl.reset_episode()

            async_controller = AsyncChunkController(policy, config.prompt, config, dagger_ctrl)
            async_controller.start()

            with torch.inference_mode():
                pre_action = np.asarray(config.reset_action, dtype=np.float32)
                raw_obs, policy_observation = update_observation(auto_config.camera_names, operator)
                async_controller.update_observation(policy_observation)
                if not async_controller.wait_for_actions(config.initial_action_wait_s):
                    logger.warning(
                        "Timed out waiting for the first action chunk after %.1fs. Falling back to hold-last-action until ready.",
                        config.initial_action_wait_s,
                    )

                t = 0
                reset_requested = False
                queue_starved_logged = False

                try:
                    while t < config.max_steps:
                        if dagger_ctrl and dagger_ctrl.should_quit:
                            break
                        if keyboard_listener:
                            if keyboard_listener.check_reset():
                                logger.info("Resetting episode...")
                                reset_requested = True
                                break
                            if keyboard_listener.check_quit():
                                logger.info("Quitting...")
                                break

                        if not dagger_ctrl or dagger_ctrl.mode == DaggerMode.INFERENCE:
                            raw_obs, policy_observation = update_observation(auto_config.camera_names, operator)
                            async_controller.update_observation(policy_observation)

                            action = async_controller.pop_action()
                            if action is None:
                                action = pre_action.copy()
                                if not queue_starved_logged:
                                    logger.warning("Action buffer empty. Holding the last commanded action until a new chunk arrives.")
                                    queue_starved_logged = True
                            else:
                                queue_starved_logged = False

                            if dagger_ctrl:
                                dagger_ctrl.count_step(intervention=False)

                            if config.interpolate:
                                interp_actions = interpolate_action(config.step_length, pre_action, action)
                            else:
                                interp_actions = action[np.newaxis, :]

                            for act in interp_actions:
                                operator.send_action(act)
                                time.sleep(1.0 / config.step_rate)

                            async_controller.mark_step_executed()
                            t += 1
                            pre_action = action.copy()

                        elif dagger_ctrl.mode == DaggerMode.ALIGNING:
                            logger.info("[DAgger] Starting alignment...")
                            async_controller.clear_actions()
                            dagger_ctrl.execute_alignment(
                                get_leader_qpos=operator.get_leader_qpos,
                                get_follower_qpos=operator.get_follower_qpos,
                                send_leader_action=operator.send_leader_action,
                                switch_leader_mode_sampling=lambda: operator.switch_leader_mode(SystemMode.SAMPLING),
                                switch_leader_mode_passive=lambda: operator.switch_leader_mode(SystemMode.PASSIVE),
                            )

                        elif dagger_ctrl.mode == DaggerMode.DEMONSTRATING:
                            raw_obs, _policy_observation = update_observation(auto_config.camera_names, operator)
                            leader_qpos = operator.get_leader_qpos()
                            operator.send_action(leader_qpos)
                            dagger_ctrl.count_step(intervention=True)
                            time.sleep(1.0 / config.step_rate)
                            t += 1
                            pre_action = leader_qpos.copy()

                        elif dagger_ctrl.mode == DaggerMode.RESUMING:
                            logger.info("[DAgger] Resuming inference...")
                            async_controller.clear_actions()

                            def _home_leaders():
                                try:
                                    operator.switch_leader_mode(SystemMode.RESETTING)
                                    operator.send_leader_action(np.asarray(config.reset_action, dtype=np.float32))
                                    time.sleep(1.0)
                                    operator.switch_leader_mode(SystemMode.PASSIVE)
                                except Exception as exc:
                                    logger.warning("[DAgger] Leader homing failed: %s", exc)

                            threading.Thread(target=_home_leaders, daemon=True).start()
                            dagger_ctrl.complete_resume()
                            raw_obs, policy_observation = update_observation(auto_config.camera_names, operator)
                            async_controller.update_observation(policy_observation)

                finally:
                    async_controller.shutdown()

                if keyboard_listener and keyboard_listener.check_quit():
                    break

                if not reset_requested and not (dagger_ctrl and dagger_ctrl.should_quit):
                    logger.info("Episode completed.")
    finally:
        if keyboard_listener:
            keyboard_listener.stop()
        if dagger_ctrl:
            dagger_ctrl.shutdown()
        operator.shutdown()


def main():
    model_inference(config, Robot(config.robot_config))


if __name__ == "__main__":
    main()
