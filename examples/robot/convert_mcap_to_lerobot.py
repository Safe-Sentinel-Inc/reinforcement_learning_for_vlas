"""Convert an MCAP dataset into a LeRobot-compatible dataset.

Usage:
    HF_LEROBOT_HOME=./lerobot_data python examples/robot/convert_mcap_to_lerobot.py --data_dir mcap_data

The target directory must include a ``config.py`` that declares topic names,
camera mappings, and other dataset metadata. See ``mcap_data/config.py`` for
a working example.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import flatbuffers
import flatbuffers.number_types
import flatbuffers.table
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
import numpy as np
import tyro

from openpi_client import image_tools

@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Metadata describing the layout and labeling of an MCAP dataset."""
    task_name: str
    robot_type: str
    folders: list[str]
    state_topics: list[str]
    action_topics: list[str]
    camera_topics: dict[str, str]
    fps: int
    delta_action_mask: tuple[int, ...] = ()
    # Labeling metadata consumed by the return-labeling scripts
    success_episodes: str | list[int] = "all"
    failed_episodes: tuple[int, ...] = ()
    all_human: bool = False
    intervention_episodes: dict = dataclasses.field(default_factory=dict)
    stage_boundaries: tuple[int, ...] = ()


def load_task_config(config_path: str | Path) -> TaskConfig:
    """Import a Python config file and return its contents as a TaskConfig."""
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    spec = importlib.util.spec_from_file_location("_mcap_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_mcap_cfg"] = mod
    spec.loader.exec_module(mod)
    return TaskConfig(
        task_name=mod.TASK_NAME,
        robot_type=mod.ROBOT_TYPE,
        folders=mod.FOLDERS,
        state_topics=mod.STATE_TOPICS,
        action_topics=mod.ACTION_TOPICS,
        camera_topics=mod.CAMERA_TOPICS,
        fps=mod.FPS,
        delta_action_mask=tuple(getattr(mod, "DELTA_ACTION_MASK", ())),
        success_episodes=getattr(mod, "SUCCESS_EPISODES", "all"),
        failed_episodes=tuple(getattr(mod, "FAILED_EPISODES", ())),
        all_human=getattr(mod, "ALL_HUMAN", False),
        intervention_episodes=dict(getattr(mod, "INTERVENTION_EPISODES", {})),
        stage_boundaries=tuple(getattr(mod, "STAGE_BOUNDARIES", ())),
    )


def decode_float_array(data: bytes) -> np.ndarray:
    """Deserialize an ``airbot_fbs.FloatArray`` FlatBuffers payload into a numpy array."""
    root_offset = flatbuffers.packer.uoffset.unpack_from(data, 0)[0]
    tab = flatbuffers.table.Table(bytearray(data), root_offset)
    o = flatbuffers.number_types.UOffsetTFlags.py_type(tab.Offset(4))
    if o != 0:
        return tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
    return np.array([], dtype=np.float32)


class McapConverter:
    """Reads MCAP recordings and writes them as a LeRobot dataset."""

    def __init__(self, data_dir: str):
        # Verify that the data directory exists
        if not Path(data_dir).is_dir():
            raise ValueError(f"data_dir '{data_dir}' is not a valid directory")
        self.data_dir = data_dir

        # Locate the dataset config file
        config_path = Path(self.data_dir) / "config.py"
        if not config_path.is_file():
            raise ValueError(f"Config file 'config.py' not found in '{data_dir}'.")

        # Parse the config and store relevant fields
        config = load_task_config(config_path)
        self.task_name = config.task_name
        self.robot_type = config.robot_type
        self.folders = config.folders
        self.state_topics = config.state_topics
        self.action_topics = config.action_topics
        self.camera_topics: dict[str:str] = config.camera_topics
        self.fps = config.fps
        self.target_height = 224
        self.target_width = 224

        # Walk each configured folder and tally MCAP files
        self.folder_file_counts = {}
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            if not folder_path.is_dir():
                raise ValueError(f"Configured folder '{folder}' not found in '{self.data_dir}'.")

            # Count recordings in this subfolder
            mcap_files = [
                f for f in os.listdir(folder_path) if Path(folder_path / f).is_file() and f.lower().endswith(".mcap")
            ]
            self.folder_file_counts[folder] = len(mcap_files)

        print(f"Found {sum(self.folder_file_counts.values())} MCAP files across {len(self.folders)} folders.")

        # Inspect a sample file to learn schema and image dimensions
        self.schemas = {}
        self.schemas["airbot_fbs.FloatArray"] = decode_float_array
        self.state_length = 0
        self.action_length = 0
        self.camera_image_shape = {}

        # Locate the first available MCAP file for introspection
        mcap_file_path = None
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(".mcap"):
                        mcap_file_path = Path(root) / filename
                        break
                if mcap_file_path:
                    break

        if mcap_file_path is None:
            raise ValueError("No MCAP files found in any of the configured folders.")

        print(f"Using {mcap_file_path} to read schema and camera information.")

        with mcap_file_path.open("rb") as f:
            reader = make_reader(f)

            # Extract camera frame dimensions from video attachments
            cam_attachment_path = {}
            for attach in reader.iter_attachments():
                media_type = attach.media_type
                if media_type == "video/mp4" and attach.name in self.camera_topics.values():
                    cam_attachment_path[attach.name] = self._save_temporary_video(attach.data)

            for camera_name, topic in self.camera_topics.items():
                if topic in cam_attachment_path:
                    frame = self._get_frame_image(cam_attachment_path[topic], 0)
                    self.camera_image_shape[camera_name] = frame.shape
                else:
                    raise ValueError(f"Camera attachment for {camera_name} not found in {mcap_file_path}.")

            print(f"Camera image shapes: {self.camera_image_shape}")

            is_read_topics = {topic: False for topic in self.state_topics + self.action_topics}
            for schema_obj, channel_obj, message_obj in reader.iter_messages(
                topics=self.state_topics + self.action_topics
            ):
                if is_read_topics[channel_obj.topic]:
                    break
                is_read_topics[channel_obj.topic] = True
                if schema_obj.name not in self.schemas:
                    raise ValueError(f"Schema '{schema_obj.name}' not found in schemas.")
                if channel_obj.topic in self.state_topics:
                    self.state_length += len(self.schemas[schema_obj.name](message_obj.data))
                if channel_obj.topic in self.action_topics:
                    self.action_length += len(self.schemas[schema_obj.name](message_obj.data))

        print(f"State length: {self.state_length}, Action length: {self.action_length}")

    def create_dataset(self) -> LeRobotDataset:
        """Build an empty LeRobot dataset with the correct feature schema."""
        features = {}
        for camera_name in self.camera_topics:
            features[camera_name] = {
                "dtype": "image",
                "shape": (self.target_height, self.target_width, 3),
                "names": ["height", "width", "channel"],
            }
        features["state"] = {
            "dtype": "float32",
            "shape": (self.state_length,),
            "names": ["state"],
        }
        features["actions"] = {
            "dtype": "float32",
            "shape": (self.action_length,),
            "names": ["actions"],
        }
        print(f"Creating LeRobot dataset with features: {features}")

        return LeRobotDataset.create(
            repo_id=self.task_name,
            robot_type=self.robot_type,
            fps=self.fps,
            features=features,
            image_writer_threads=10,  # TODO: add config for image writer threads
            image_writer_processes=5,  # TODO: add config for image writer processes
        )

    def validate_metadata(self, mcap_file_path: Path) -> bool:
        """Check that the task name in the MCAP metadata matches expectations."""
        try:
            with mcap_file_path.open("rb") as f:
                reader = make_reader(f)
                for md in reader.iter_metadata():
                    if md.name == "task_info":
                        task_name = md.metadata.get("task_name", "N/A")
                        # Strip stray quotes that sometimes appear in metadata values
                        task_name_clean = task_name.strip('"').strip("'")
                        if task_name_clean != self.task_name:
                            print(f"Task name mismatch: expected {self.task_name}, got {task_name} (cleaned: {task_name_clean})")
                            return False
            return True
        except Exception as e:
            print(f"Error reading MCAP file {mcap_file_path}: {e}")
            return False

    def _save_temporary_video(self, bytes_data: bytes) -> str:
        """Write raw video bytes from an MCAP attachment to a temporary file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_file.flush()
        return tmp_file.name

    @staticmethod
    def _get_frame_image(path: str, frame_index: int) -> np.ndarray:
        """Decode and return a single frame from a video file by index."""
        cap = cv2.VideoCapture(path)
        frame = None
        for i in range(frame_index + 1):
            ret, new_frame = cap.read()
            if not ret:
                if i <= frame_index:
                    print(
                        f"Warning: Requested frame_index {frame_index} not found in video {path} until {i}. Using last frame instead."
                    )
                break
            frame = new_frame
        cap.release()
        if frame is None:
            raise ValueError(f"Failed to read frame {frame_index} from video {path}.")
        # LeRobot expects RGB channel order
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def load_to_dataset(self, dataset: LeRobotDataset):
        """Iterate over all MCAP files and append their episodes to the dataset."""
        for folder_name, file_count in self.folder_file_counts.items():
            print(f"Processing folder: {folder_name} with {file_count} files")

            for root, _, files in os.walk(Path(self.data_dir) / folder_name):
                for filename in files:
                    if not filename.lower().endswith(".mcap"):
                        continue
                    mcap_file_path = Path(root) / filename
                    # Ensure the file metadata is consistent
                    if not self.validate_metadata(mcap_file_path):
                        print(f"Metadata validation failed for {folder_name} file {filename}. Skipping this file.")
                        continue

                    print(f"Loading and converting {folder_name} file {filename} to LeRobot format...")
                    
                    # Keep video decoders open for the lifetime of this file
                    caps = {}
                    last_frames = {}  # fallback frame when video ends early

                    try:
                        with mcap_file_path.open("rb") as f:
                            reader = make_reader(f)
                            cnt_topics = {topic: 0 for topic in self.state_topics + self.action_topics}
                            state_msg = {}
                            action_msg = {}
                            cnt = 0
                            cam_attachment_path = {}
                            for attach in reader.iter_attachments():
                                media_type = attach.media_type
                                if media_type == "video/mp4" and attach.name in self.camera_topics.values():
                                    cam_attachment_path[attach.name] = self._save_temporary_video(attach.data)
                            if not cam_attachment_path or len(cam_attachment_path) != len(self.camera_topics):
                                print(
                                    f"Warning: Not all camera attachments were found in {mcap_file_path}. Expected {len(self.camera_topics)}, found {len(cam_attachment_path)}."
                                )
                                continue
                                
                            # Open one VideoCapture per camera topic
                            for topic, path in cam_attachment_path.items():
                                caps[topic] = cv2.VideoCapture(path)
                                last_frames[topic] = None

                            for schema_obj, channel_obj, message_obj in reader.iter_messages(
                                topics=self.state_topics + self.action_topics
                            ):
                                cnt_topics[channel_obj.topic] += 1
                                if cnt_topics[channel_obj.topic] - cnt == 2:
                                    self._save_frame(dataset, state_msg, action_msg, caps, last_frames, cnt)
                                    cnt += 1
                                if channel_obj.topic in self.state_topics:
                                    state_msg[channel_obj.topic] = self.schemas[schema_obj.name](
                                        message_obj.data
                                    )
                                if channel_obj.topic in self.action_topics:
                                    action_msg[channel_obj.topic] = self.schemas[schema_obj.name](
                                        message_obj.data
                                    )
                            # Flush the final frame of the episode
                            self._save_frame(dataset, state_msg, action_msg, caps, last_frames, cnt)

                            dataset.save_episode()

                    finally:
                        # Close video decoders
                        for cap in caps.values():
                            cap.release()
                        # Remove temporary video files from disk
                        for temp_path in cam_attachment_path.values():
                            try:
                                Path(temp_path).unlink(missing_ok=True)
                            except Exception as e:
                                print(f"Warning: Could not delete temporary file {temp_path}: {e}")

    def _save_frame(
        self, dataset: LeRobotDataset, state_msg: dict, action_msg: dict, caps: dict, last_frames: dict, cnt: int
    ):
        """Assemble one timestep from state, action, and camera data and add it to the dataset."""
        # Skip this frame if any required topic is missing
        for topic in self.state_topics:
            if topic not in state_msg:
                print(f"Warning: State topic {topic} not found in messages. Skipping frame {cnt}.")
                return
        for topic in self.action_topics:
            if topic not in action_msg:
                print(f"Warning: Action topic {topic} not found in messages. Skipping frame {cnt}.")
                return

        state_vec = np.array([])
        action_vec = np.array([])
        for topic in self.state_topics:
            state_vec = np.concatenate((state_vec, state_msg[topic]))
        for topic in self.action_topics:
            action_vec = np.concatenate((action_vec, action_msg[topic]))
        frame = {}
        for camera_name, topic in self.camera_topics.items():
            if topic in caps:
                # Read the next sequential frame from the open video decoder
                ret, img = caps[topic].read()
                if ret:
                    # Valid frame obtained
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    last_frames[topic] = img
                elif last_frames[topic] is not None:
                    # Video ran out of frames; fall back to the last good one
                    img = last_frames[topic]
                else:
                    # No frame was ever read for this camera; use a black image
                    print(f"Warning: Failed to read *any* frames for {camera_name}. Using black frame.")
                    img = np.zeros(self.camera_image_shape[camera_name], dtype=np.uint8)
                
                img = image_tools.resize_with_pad(img, self.target_height, self.target_width)
                frame[camera_name] = img
            else:
                print(f"Warning: Camera attachment for {camera_name} not found.")

        frame["state"] = state_vec.astype(np.float32)
        frame["actions"] = action_vec.astype(np.float32)
        frame["task"] = self.task_name
        dataset.add_frame(frame)


def main(data_dir: str):
    mcap_converter = McapConverter(data_dir)

    # Remove a previous dataset at the output path if present
    output_path = HF_LEROBOT_HOME / mcap_converter.task_name
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create the dataset and populate it with all episodes
    dataset = mcap_converter.create_dataset()
    mcap_converter.load_to_dataset(dataset)


if __name__ == "__main__":
    tyro.cli(main)
