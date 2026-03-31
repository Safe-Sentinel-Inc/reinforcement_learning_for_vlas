from airbot_ie.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig
from airdc.common.devices.cameras.v4l2 import V4L2Camera, V4L2CameraConfig
from airdc.common.systems.basis import SystemMode
from robot_config import RobotConfig

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Robot:
    """Hardware interface for the Play dual-arm robot platform."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.robots = {
            name: AIRBOTPlay(AIRBOTPlayConfig(port=port))
            for name, port in zip(self.config.robot_groups, self.config.robot_ports, strict=True)
        }
        self.cameras = {
            name: V4L2Camera(V4L2CameraConfig(camera_index=index))
            for name, index in zip(self.config.camera_names, self.config.camera_index, strict=True)
        }
        self.keys = list(self.robots.keys()) + list(self.cameras.keys())
        self.values = list(self.robots.values()) + list(self.cameras.values())
        for key, value in zip(self.keys, self.values, strict=True):
            if not value.configure():
                raise RuntimeError(f"Failed to configure {key}.")

        # Leader arms are created on demand when DAgger mode is activated
        self.leaders: dict[str, AIRBOTPlay] = {}

    def init_leaders(self):
        """Connect to the leader arms for human demonstration.

        Each leader arm is paired with the follower in the same robot group
        via the port numbers specified in RobotConfig.leader_ports.
        """
        if self.leaders:
            logger.info("Leader arms already initialized.")
            return

        for name, port in zip(self.config.robot_groups, self.config.leader_ports, strict=True):
            leader = AIRBOTPlay(AIRBOTPlayConfig(port=port))
            if not leader.configure():
                raise RuntimeError(f"Failed to configure leader arm '{name}' on port {port}.")
            self.leaders[name] = leader
            logger.info(f"Leader arm '{name}' initialized on port {port}.")

        # Begin in gravity-compensation mode so the arms hang freely
        self.switch_leader_mode(SystemMode.PASSIVE)
        logger.info("All leader arms initialized in PASSIVE mode.")

    def switch_mode(self, mode):
        """Change the operating mode of all follower arms."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def switch_follower_mode(self, mode: SystemMode):
        """Explicitly set the mode on all follower arms."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def switch_leader_mode(self, mode: SystemMode):
        """Set the operating mode on all leader arms.

        PASSIVE lets the human move the arm freely (gravity compensation).
        SAMPLING makes the arm track commanded positions.
        """
        for leader in self.leaders.values():
            leader.switch_mode(mode)

    def capture_observation(self) -> dict:
        """Read joint states and camera images from all devices."""
        obs = {}
        for name, ins in zip(self.keys, self.values, strict=True):
            for key, value in ins.capture_observation().items():
                full_key = f"{name}/{key}"
                # Camera hardware returns BGR; convert to RGB for the policy
                if "image" in key and isinstance(value.get("data"), np.ndarray):
                    image_data = value["data"]
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        value = value.copy()
                        value["data"] = image_data[..., ::-1]
                obs[full_key] = value
        return obs

    def send_action(self, action):
        """Command each follower arm with a 7-DOF joint target."""
        for index, (_group, robot) in enumerate(self.robots.items()):
            joint_target = [float(v) for v in action[index * 7 : (index + 1) * 7]]
            robot.send_action(joint_target)

    def send_leader_action(self, action):
        """Send joint targets to the leader arms.

        Typically called during alignment to move the leaders toward
        the follower positions. Requires SAMPLING mode.
        """
        for index, (_group, leader) in enumerate(self.leaders.items()):
            joint_target = [float(v) for v in action[index * 7 : (index + 1) * 7]]
            leader.send_action(joint_target)

    def get_qpos(self, obs: dict) -> list[float]:
        """Extract the full joint-position vector from a raw observation."""
        qpos = []
        for group in self.config.robot_groups:
            qpos.extend(obs[f"{group}/arm/joint_state/position"]["data"])
            qpos.extend(obs[f"{group}/eef/joint_state/position"]["data"])
        return qpos

    def get_follower_qpos(self) -> np.ndarray:
        """Read the current joint positions of all follower arms."""
        qpos = []
        for group in self.config.robot_groups:
            obs = self.robots[group].capture_observation()
            qpos.extend(obs["arm/joint_state/position"]["data"])
            qpos.extend(obs["eef/joint_state/position"]["data"])
        return np.array(qpos)

    def get_leader_qpos(self) -> np.ndarray:
        """Read the current joint positions of all leader arms."""
        qpos = []
        for group in self.config.robot_groups:
            obs = self.leaders[group].capture_observation()
            qpos.extend(obs["arm/joint_state/position"]["data"])
            qpos.extend(obs["eef/joint_state/position"]["data"])
        return np.array(qpos)

    def shutdown(self) -> bool:
        """Disconnect from all robot arms and cameras."""
        for robot in self.robots.values():
            robot.shutdown()
        for leader in self.leaders.values():
            leader.shutdown()
        for camera in self.cameras.values():
            camera.shutdown()
        return True
