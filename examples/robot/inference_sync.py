import logging
import threading
import time

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
from inference_recorder import InferenceRecorder
from inference_recorder import RecordConfig
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
    """Parameters for the synchronous inference loop.

    Args:
        max_steps: Upper bound on the number of steps per episode.
        step_rate: Control-loop frequency in Hz.
        step_length: Per-joint maximum displacement allowed in a single timestep.
        reset_action: Joint configuration the robot returns to between episodes.
        interpolate: Whether to linearly subdivide large joint-space jumps.
    """

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
    debug: bool = False
    prompt: str = ""
    record: RecordConfig = RecordConfig(record_data=False)
    dagger: DaggerConfig = DaggerConfig(enable=False)
    robot_config: RobotConfig


auto_config = AutoConfig()

config = tyro.cli(InferConfig, config=[tyro.conf.ConsolidateSubcommandArgs])
robot_config = config.robot_config
if robot_config.robot_type == "play":
    from play_operator import Robot
else:
    raise ValueError("Unsupported robot type. Please use a valid config path for the robot.")


def update_observation(camera_names: list[str], operator: System) -> dict:
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    for camera_name in camera_names:
        image_dict[camera_name] = obs[f"{camera_name}/color/image_raw"]["data"]
    auto_config.observation = {"qpos": np.array(qpos), "images": image_dict}
    return obs


def inference_once(policy, prompt: str) -> np.ndarray:
    """Query the policy server for one action chunk.

    The server handles any necessary output transforms (e.g. delta-to-absolute
    conversion), so the returned array contains joint angles that can be
    sent directly to the robot.
    """
    obs = {"state": auto_config.observation["qpos"], "prompt": prompt, "advantage": True} | auto_config.observation["images"]

    action_chunk = policy.infer(obs)["actions"] if policy is not None else np.zeros([64, 7], dtype=np.float32)
    auto_config.chunk_size_predict = action_chunk.shape[0]
    auto_config.state_dim = action_chunk.shape[1]
    return action_chunk


def model_inference(config: InferConfig, operator: System):
    auto_config.camera_names = operator.config.camera_names
    assert config.prompt, "Prompt must be provided for policy inference."

    from openpi_client import websocket_client_policy

    # Set up the websocket connection to the policy server
    policy_config = config.policy_config
    logger.info(f"Connecting to policy server at {policy_config.host}:{policy_config.port}")
    policy = websocket_client_policy.WebsocketClientPolicy(host=policy_config.host, port=policy_config.port)

    # Prepare the data recorder for saving episodes
    record_config = config.record
    if record_config.record_data and not record_config.task_name:
        record_config = record_config.model_copy(update={"task_name": config.prompt})
    recorder = InferenceRecorder(record_config, auto_config.camera_names)

    # Optionally set up the DAgger controller for human intervention
    dagger_ctrl = None
    if config.dagger.enable:
        dagger_ctrl = DaggerController(config.dagger)
        operator.init_leaders()
        dagger_ctrl.start_keyboard_listener()
        logger.info(
            "[DAgger] DAgger mode ENABLED. "
            f"Press '{config.dagger.key_enter_dagger}' to intervene, "
            f"'{config.dagger.key_resume_inference}' to resume, 'q' to quit."
        )

    # Keyboard listener for episode control when DAgger is not active
    keyboard_listener = None
    if not dagger_ctrl:
        keyboard_listener = KeyboardListener()
        keyboard_listener.start()

    try:
        while True:
            # Exit if requested
            if dagger_ctrl and dagger_ctrl.should_quit:
                logger.info("DAgger quit requested.")
                break
            if keyboard_listener and keyboard_listener.check_quit():
                logger.info("Quitting...")
                break

            # Move the robot to its home pose
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
            # Begin recording the new episode
            recorder.start_episode()
            if dagger_ctrl:
                dagger_ctrl.reset_episode()

            # Use the reset pose as the first "previous action"
            with torch.inference_mode():
                pre_action = np.array(config.reset_action)
                raw_obs = update_observation(auto_config.camera_names, operator)
                action_chunk = None
                t = 0
                reset_requested = False

                while t < config.max_steps:
                    # Handle quit and reset requests
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

                    # Normal policy execution path
                    if not dagger_ctrl or dagger_ctrl.mode == DaggerMode.INFERENCE:
                        # Read the latest sensor data
                        raw_obs = update_observation(auto_config.camera_names, operator)

                        # Run a new inference when the current chunk is exhausted
                        action_index = t % config.chunk_size_execute
                        if action_index == 0:
                            start_time = time.monotonic()
                            logger.info("Start inference...")
                            action_chunk = inference_once(policy, config.prompt).copy()
                            logger.info(f"Inference time: {time.monotonic() - start_time} s")

                        action: np.ndarray = action_chunk[action_index]

                        # Log the observation and action for later replay
                        recorder.record_step(raw_obs, action, intervention=0)
                        if dagger_ctrl:
                            dagger_ctrl.count_step(intervention=False)

                        # Optionally subdivide large joint-space motions
                        if config.interpolate:
                            interp_actions = interpolate_action(config.step_length, pre_action, action)
                        else:
                            interp_actions = action[np.newaxis, :]
                        # Send each sub-step to the robot at the control rate
                        for act in interp_actions:
                            operator.send_action(act)
                            time.sleep(1.0 / config.step_rate)
                        t += 1
                        pre_action = action.copy()

                    # Leader-to-follower alignment phase
                    elif dagger_ctrl.mode == DaggerMode.ALIGNING:
                        logger.info("[DAgger] Starting alignment...")
                        dagger_ctrl.execute_alignment(
                            get_leader_qpos=operator.get_leader_qpos,
                            get_follower_qpos=operator.get_follower_qpos,
                            send_leader_action=operator.send_leader_action,
                            switch_leader_mode_sampling=lambda: operator.switch_leader_mode(SystemMode.SAMPLING),
                            switch_leader_mode_passive=lambda: operator.switch_leader_mode(SystemMode.PASSIVE),
                        )

                    # Human demonstration phase
                    elif dagger_ctrl.mode == DaggerMode.DEMONSTRATING:
                        # Grab sensor data from followers and cameras
                        raw_obs = update_observation(auto_config.camera_names, operator)

                        # Use the leader joint positions as the action
                        leader_qpos = operator.get_leader_qpos()

                        # Command followers to mirror the leader
                        operator.send_action(leader_qpos)

                        # Mark this step as a human intervention
                        recorder.record_step(raw_obs, leader_qpos, intervention=1)
                        dagger_ctrl.count_step(intervention=True)

                        time.sleep(1.0 / config.step_rate)
                        t += 1
                        pre_action = leader_qpos.copy()

                    # Transition back to autonomous control
                    elif dagger_ctrl.mode == DaggerMode.RESUMING:
                        logger.info("[DAgger] Resuming inference...")

                        # Return leader arms to home in a background thread
                        def _home_leaders():
                            try:
                                operator.switch_leader_mode(SystemMode.RESETTING)
                                operator.send_leader_action(np.array(config.reset_action))
                                time.sleep(1.0)
                                operator.switch_leader_mode(SystemMode.PASSIVE)
                            except Exception as e:
                                logger.warning(f"[DAgger] Leader homing failed: {e}")

                        threading.Thread(target=_home_leaders, daemon=True).start()

                        # Advance the step counter so a fresh inference runs immediately
                        t = (t // config.chunk_size_execute + 1) * config.chunk_size_execute

                        # Finalize the mode switch
                        dagger_ctrl.complete_resume()

                # Persist the episode data to disk
                dagger_stats = dagger_ctrl.stats.to_dict() if dagger_ctrl else None
                recorder.save_episode(dagger_stats=dagger_stats)

                if keyboard_listener and keyboard_listener.check_quit():
                    break

                if not reset_requested and not (dagger_ctrl and dagger_ctrl.should_quit):
                    logger.info("Episode completed.")
    finally:
        if keyboard_listener:
            keyboard_listener.stop()
        if dagger_ctrl:
            dagger_ctrl.shutdown()
        recorder.shutdown()
        operator.shutdown()


def main():
    model_inference(config, Robot(config.robot_config))


if __name__ == "__main__":
    main()
