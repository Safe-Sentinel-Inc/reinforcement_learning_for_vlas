"""Records observations and actions during policy inference to MCAP files.

The MCAP output matches the schema used by the data-collection pipeline,
so recorded episodes can be converted to LeRobot format with the same
conversion scripts used for teleoperation data.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from airdc.common.samplers.mcap_sampler import McapDataSampler, McapDataSamplerConfig
from airdc.common.samplers.basis import SaveType, TaskInfo

logger = logging.getLogger(__name__)


class RecordConfig(BaseModel):
    """Settings for saving inference data to disk.

    Args:
        record_data: Enable or disable recording.
        save_dir: Destination folder for MCAP files.
        task_name: Task label embedded in MCAP metadata (defaults to the prompt).
        save_video: Video codec for camera streams ("h264", "jpeg", or "raw").
    """

    record_data: bool = True
    save_dir: str = "./inference_data"  # default directory, can be changed
    task_name: str = ""
    save_video: str = "h264"


class InferenceRecorder:
    """Writes per-step observations and actions into MCAP files.

    Internally delegates to the same McapDataSampler used during teleoperation,
    so each episode produces a file that is directly compatible with the
    MCAP-to-LeRobot conversion pipeline. Recorded contents include joint
    states, camera frames (H264-encoded by default), predicted actions,
    and per-step timestamps.
    """

    def __init__(self, config: RecordConfig, camera_names: list[str]):
        """Set up the recorder.

        Args:
            config: Recording settings.
            camera_names: Camera identifiers matching the robot operator.
        """
        self._config = config
        self._camera_names = camera_names
        self._sampler: Optional[McapDataSampler] = None
        self._round: int = 0
        self._step_count: int = 0
        self._recording: bool = False
        self._round_data: dict = defaultdict(list)

        if not config.record_data:
            logger.info("Data recording is disabled.")
            return

        # Ensure the output directory exists
        self._save_dir = Path(config.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # Resume numbering from the count of files already present
        existing_files = list(self._save_dir.glob("*.mcap"))
        self._round = len(existing_files)

        # Set up the MCAP writer
        sampler_config = McapDataSamplerConfig(
            task_info=TaskInfo(task_name=config.task_name),
            save_type=SaveType(color=config.save_video),
        )
        self._sampler = McapDataSampler(sampler_config)
        if not self._sampler.configure():
            logger.error("Failed to configure McapDataSampler.")
            self._sampler = None
            return

        # Attach metadata to the sampler
        self._sampler.set_info({"recorder": "inference_recorder"})
        logger.info(
            f"Inference recorder initialized. Save dir: {self._save_dir}, "
            f"starting round: {self._round}, format: {config.save_video}"
        )

    @property
    def enabled(self) -> bool:
        """True when recording is enabled and the sampler is ready."""
        return self._config.record_data and self._sampler is not None

    def start_episode(self) -> None:
        """Open a fresh MCAP file for the next episode."""
        if not self.enabled:
            return

        try:
            self._save_path = self._sampler.compose_path(self._save_dir, self._round)
            self._step_count = 0
            self._round_data = defaultdict(list)
            self._recording = True
            logger.info(f"Started recording episode {self._round} -> {self._save_path}")
        except Exception as e:
            logger.error(f"Failed to start episode recording: {e}")
            self._recording = False

    def record_step(self, raw_obs: dict, action: np.ndarray, intervention: int = 0) -> None:
        """Append one timestep of data to the current episode.

        The raw observation and the executed action are converted to the
        MCAP topic layout expected by the conversion pipeline.

        Args:
            raw_obs: Observation dict returned by operator.capture_observation().
            action: Joint-space action that was executed this step.
            intervention: 0 for policy-generated, 1 for human-demonstrated.
        """
        if not self._recording:
            return

        try:
            log_stamp = time.time_ns()

            # Assemble the data dictionary in the expected MCAP layout
            data = {}

            # Prefix keys with "/" to conform to MCAP topic naming
            for key, value in raw_obs.items():
                topic_key = f"/{key}" if not key.startswith("/") else key
                data[topic_key] = value

            # Store the predicted action under "lead" topics to match teleoperation format
            self._add_action_as_lead(data, raw_obs, action)

            # Include the intervention marker
            data["/dagger/intervention"] = {
                "data": [intervention],
                "t": log_stamp,
            }

            # Attach the timestamp
            data["log_stamps"] = log_stamp

            # Let the sampler encode images and serialize messages
            processed = self._sampler.update(data)

            # Buffer processed data for the end-of-episode save
            for key, value in processed.items():
                self._round_data[key].append(value)

            self._step_count += 1

        except Exception as e:
            logger.warning(f"Failed to record step {self._step_count}: {e}")

    def _add_action_as_lead(
        self, data: dict, raw_obs: dict, action: np.ndarray
    ) -> None:
        """Map the action array onto leader-side MCAP topics.

        The action vector is split into arm and end-effector segments and
        written under "lead" topics that mirror the corresponding follower
        observation topics, keeping the file compatible with teleoperation
        recordings.

        Args:
            data: Step data dictionary being assembled.
            raw_obs: Raw observation used to detect the robot layout.
            action: Action array to store.
        """
        stamp = time.time_ns()

        # Determine the arm layout from observation key prefixes
        obs_keys = list(raw_obs.keys())

        # Identify whether this is a dual-arm setup
        has_left = any(k.startswith("left/") for k in obs_keys)
        has_right = any(k.startswith("right/") for k in obs_keys)

        if has_left and has_right:
            # Dual-arm layout: split into left and right arm + gripper segments
            action_flat = np.array(action).flatten()
            left_arm_action = action_flat[:6].tolist()
            left_eef_action = action_flat[6:7].tolist()
            right_arm_action = action_flat[7:13].tolist()
            right_eef_action = action_flat[13:14].tolist()

            data["/left/lead/arm/joint_state/position"] = {
                "data": left_arm_action,
                "t": stamp,
            }
            data["/left/lead/eef/joint_state/position"] = {
                "data": left_eef_action,
                "t": stamp,
            }
            data["/right/lead/arm/joint_state/position"] = {
                "data": right_arm_action,
                "t": stamp,
            }
            data["/right/lead/eef/joint_state/position"] = {
                "data": right_eef_action,
                "t": stamp,
            }
        else:
            # Single-arm layout: one arm segment plus gripper
            action_flat = np.array(action).flatten()
            if len(action_flat) >= 7:
                arm_action = action_flat[:6].tolist()
                eef_action = action_flat[6:7].tolist()
                data["/lead/arm/joint_state/position"] = {
                    "data": arm_action,
                    "t": stamp,
                }
                data["/lead/eef/joint_state/position"] = {
                    "data": eef_action,
                    "t": stamp,
                }
            else:
                data["/lead/action"] = {
                    "data": action_flat.tolist(),
                    "t": stamp,
                }

    def save_episode(self, dagger_stats: dict = None) -> Optional[Path]:
        """Flush buffered data to the MCAP file and close the episode.

        Args:
            dagger_stats: Optional intervention statistics to include in the log.

        Returns:
            Path to the written file, or None on failure.
        """
        if not self._recording:
            return None

        try:
            self._recording = False
            result_path = self._sampler.save(self._save_path, self._round_data)
            self._round += 1
            self._round_data = defaultdict(list)

            stats_msg = ""
            if dagger_stats:
                stats_msg = (
                    f", interventions: {dagger_stats.get('total_interventions', 0)}, "
                    f"ratio: {dagger_stats.get('intervention_ratio', 0):.1%}"
                )
            logger.info(
                f"Saved episode to {result_path} ({self._step_count} steps{stats_msg})"
            )
            return Path(result_path)
        except Exception as e:
            logger.error(f"Failed to save episode: {e}")
            return None

    def shutdown(self) -> None:
        """Save any in-progress episode and release the MCAP writer."""
        if self._recording:
            logger.info("Saving in-progress episode before shutdown...")
            self.save_episode()

        if self._sampler is not None:
            self._sampler.shutdown()
            logger.info("Inference recorder shut down.")
