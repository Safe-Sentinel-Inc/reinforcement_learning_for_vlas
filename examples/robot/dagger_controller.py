"""DAgger controller for human intervention during autonomous policy execution.

Provides a Dataset Aggregation (DAgger) mechanism that lets a human operator
seamlessly take over control mid-episode and hand it back to the policy:
  - Press 'i' to pause the policy and enter demonstration mode.
  - Leader arms are smoothly moved to match follower positions via cosine interpolation.
  - The human demonstrates corrective behavior through leader-arm teleoperation.
  - Press 'o' to hand control back to the policy.
"""

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DaggerMode(Enum):
    """Possible states of the DAgger state machine."""
    INFERENCE = "inference"
    ALIGNING = "aligning"
    DEMONSTRATING = "demonstrating"
    RESUMING = "resuming"


class DaggerConfig(BaseModel):
    """Settings that control DAgger behaviour.

    Args:
        enable: Turn DAgger on or off.
        key_enter_dagger: Keyboard key that switches to human control.
        key_resume_inference: Keyboard key that returns to policy control.
        align_steps: Number of waypoints in the leader-to-follower alignment motion.
        align_duration: Time in seconds for the full alignment motion.
        gripper_threshold: Value above which the gripper is considered open.
    """
    enable: bool = True
    key_enter_dagger: str = "i"
    key_resume_inference: str = "o"
    align_steps: int = 50
    align_duration: float = 1.0
    gripper_threshold: float = 0.5


@dataclass
class DaggerStats:
    """Tracks intervention counts and step-level statistics for a single episode."""
    total_interventions: int = 0
    intervention_steps: int = 0
    autonomous_steps: int = 0
    segments: list = field(default_factory=list)
    _current_segment_start: int = field(default=0, repr=False)
    _current_segment_type: str = field(default="policy", repr=False)

    def start_intervention(self, step: int):
        self._close_segment(step)
        self._current_segment_type = "intervention"
        self._current_segment_start = step
        self.total_interventions += 1

    def end_intervention(self, step: int):
        self._close_segment(step)
        self._current_segment_type = "policy"
        self._current_segment_start = step

    def _close_segment(self, step: int):
        if step > self._current_segment_start:
            self.segments.append({
                "type": self._current_segment_type,
                "start": self._current_segment_start,
                "end": step - 1,
            })

    def to_dict(self) -> dict:
        total = self.intervention_steps + self.autonomous_steps
        return {
            "total_interventions": self.total_interventions,
            "intervention_steps": self.intervention_steps,
            "autonomous_steps": self.autonomous_steps,
            "intervention_ratio": self.intervention_steps / max(1, total),
            "segments": self.segments,
        }

    def reset(self):
        self.total_interventions = 0
        self.intervention_steps = 0
        self.autonomous_steps = 0
        self.segments.clear()
        self._current_segment_start = 0
        self._current_segment_type = "policy"


class DaggerController:
    """Manages mode transitions, leader-arm alignment, and bookkeeping for DAgger.

    Threading layout:
      - A daemon keyboard thread captures raw keystrokes (no Enter needed).
      - The main loop inspects the current mode and acts accordingly.
      - A threading.Event gates whether policy inference should be paused.
    """

    def __init__(self, config: DaggerConfig):
        self.config = config

        # Mode protected by a lock for thread safety
        self._mode = DaggerMode.INFERENCE
        self._mode_lock = threading.Lock()
        self._pause_event = threading.Event()  # when set, inference is paused

        # Per-episode statistics
        self.stats = DaggerStats()
        self._step_counter = 0

        # Shutdown coordination
        self._shutdown = threading.Event()
        self._keyboard_thread: Optional[threading.Thread] = None

    @property
    def mode(self) -> DaggerMode:
        with self._mode_lock:
            return self._mode

    @mode.setter
    def mode(self, value: DaggerMode):
        with self._mode_lock:
            self._mode = value

    @property
    def is_intervention(self) -> bool:
        return self.mode in (DaggerMode.DEMONSTRATING, DaggerMode.ALIGNING)

    @property
    def inference_paused(self) -> bool:
        return self._pause_event.is_set()

    @property
    def should_quit(self) -> bool:
        return self._shutdown.is_set()

    def start_keyboard_listener(self):
        """Spawn the background thread that reads keystrokes."""
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True, name="dagger-keyboard"
        )
        self._keyboard_thread.start()
        logger.info(
            f"DAgger keyboard listener started. "
            f"Press '{self.config.key_enter_dagger}' to intervene, "
            f"'{self.config.key_resume_inference}' to resume, 'q' to quit."
        )

    def _keyboard_loop(self):
        """Read single characters from stdin in cbreak mode (Linux/macOS)."""
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._shutdown.is_set():
                ch = sys.stdin.read(1)
                if ch == self.config.key_enter_dagger:
                    self._on_enter_dagger()
                elif ch == self.config.key_resume_inference:
                    self._on_resume_inference()
                elif ch == "q":
                    logger.info("[DAgger] Quit requested.")
                    self._shutdown.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _on_enter_dagger(self):
        if self.mode != DaggerMode.INFERENCE:
            logger.warning("[DAgger] Already in DAgger mode, ignoring 'i' press.")
            return
        logger.info("[DAgger] >>> Entering DAgger mode — pausing inference...")
        self._pause_event.set()
        self.mode = DaggerMode.ALIGNING
        self.stats.start_intervention(self._step_counter)

    def _on_resume_inference(self):
        if self.mode not in (DaggerMode.DEMONSTRATING, DaggerMode.ALIGNING):
            logger.warning("[DAgger] Not in DAgger mode, ignoring 'o' press.")
            return
        logger.info("[DAgger] >>> Resuming inference mode...")
        self.mode = DaggerMode.RESUMING

    def generate_alignment_trajectory(
        self,
        leader_qpos: np.ndarray,
        follower_qpos: np.ndarray,
        dof_per_arm: int = 7,
    ) -> list[np.ndarray]:
        """Build a cosine-interpolated path that brings the leader to the follower pose.

        Gripper joints (last DOF of each arm) are not interpolated; they
        immediately adopt the follower's gripper value.

        Args:
            leader_qpos: Starting joint positions of the leader arms.
            follower_qpos: Target joint positions (current follower state).
            dof_per_arm: Joints per arm including the gripper (default 7).

        Returns:
            Ordered list of waypoint arrays forming the alignment path.
        """
        steps = self.config.align_steps
        n_joints = len(leader_qpos)
        n_arms = n_joints // dof_per_arm
        gripper_indices = [arm_idx * dof_per_arm + (dof_per_arm - 1) for arm_idx in range(n_arms)]

        trajectory = []
        for i in range(1, steps + 1):
            alpha = 0.5 * (1 - math.cos(math.pi * i / steps))
            waypoint = (1 - alpha) * leader_qpos + alpha * follower_qpos

            # Gripper values are set directly rather than interpolated
            for gi in gripper_indices:
                waypoint[gi] = follower_qpos[gi]

            trajectory.append(waypoint.copy())

        return trajectory

    def execute_alignment(
        self,
        get_leader_qpos: Callable[[], np.ndarray],
        get_follower_qpos: Callable[[], np.ndarray],
        send_leader_action: Callable[[np.ndarray], None],
        switch_leader_mode_sampling: Callable[[], None],
        switch_leader_mode_passive: Callable[[], None],
    ):
        """Run the full leader-to-follower alignment procedure.

        Steps:
        1. Put the leader into position-servo mode so it can receive commands.
        2. Execute the cosine-interpolated trajectory.
        3. Switch the leader to passive (gravity-comp) so the human can move it.
        4. Transition the state machine to DEMONSTRATING.
        """
        leader_qpos = np.array(get_leader_qpos())
        follower_qpos = np.array(get_follower_qpos())

        logger.info(f"[DAgger] Aligning leader to follower ({self.config.align_steps} steps)...")

        # Position-servo mode is required for the leader to track commands
        switch_leader_mode_sampling()

        trajectory = self.generate_alignment_trajectory(leader_qpos, follower_qpos)
        dt = self.config.align_duration / self.config.align_steps

        for waypoint in trajectory:
            if self._shutdown.is_set():
                return
            send_leader_action(waypoint)
            time.sleep(dt)

        # Gravity-comp mode allows the human to freely move the arm
        switch_leader_mode_passive()
        self.mode = DaggerMode.DEMONSTRATING
        logger.info("[DAgger] Alignment complete. Leader in PASSIVE mode — you may demonstrate now.")

    def count_step(self, intervention: bool):
        """Record one timestep as either autonomous or human-controlled."""
        self._step_counter += 1
        if intervention:
            self.stats.intervention_steps += 1
        else:
            self.stats.autonomous_steps += 1

    def complete_resume(self):
        """Finish the transition from demonstration back to policy inference."""
        self.stats.end_intervention(self._step_counter)
        self.mode = DaggerMode.INFERENCE
        self._pause_event.clear()
        logger.info("[DAgger] Inference resumed.")

    def reset_episode(self):
        """Clear all state in preparation for a new episode."""
        self.stats.reset()
        self._step_counter = 0
        self.mode = DaggerMode.INFERENCE
        self._pause_event.clear()

    def shutdown(self):
        """Signal all threads to stop and log final statistics."""
        self._shutdown.set()
        final_stats = self.stats.to_dict()
        logger.info(f"[DAgger] Session stats: {final_stats}")
        return final_stats
