from typing import Any

from pydantic import BaseModel


class RobotConfig(BaseModel):
    """Hardware layout describing arm ports, camera indices, and robot groups."""

    robot_type: str = "play"
    robot_groups: list[str] = ["left", "right"]
    robot_ports: list[int] = [50051, 50053]
    leader_ports: list[int] = [50050, 50052]
    camera_names: list[str] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    camera_index: list[int | str] = [2, 4, 6]

    def model_post_init(self, __context: Any) -> None:
        for _ in range(len(self.camera_names) - len(self.camera_index)):
            self.camera_names.pop()
