"""Waypoint-driven velocity command for the second-stage 400 m task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat


class WaypointVelocityCommand(CommandTerm):
    """Per-environment ordered waypoint tracker using no external sensors."""

    cfg: "WaypointVelocityCommandCfg"

    def __init__(self, cfg: "WaypointVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.track_points = torch.tensor(self._build_track(), dtype=torch.float32, device=self.device)
        self.next_target_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.just_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.metrics["waypoints_reached"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _build_track(self) -> list[tuple[float, float]]:
        start, radius, straight = 32.5, 23.24, 110.43
        half_curve = math.pi * radius

        def point_at(distance: float) -> tuple[float, float]:
            if distance < start:
                return distance, 0.0
            if distance < start + half_curve:
                theta = (distance - start) / radius
                return start + radius * math.sin(theta), radius * (1.0 - math.cos(theta))
            if distance < start + half_curve + straight:
                return start - (distance - start - half_curve), 2.0 * radius
            if distance < start + 2.0 * half_curve + straight:
                theta = (distance - start - half_curve - straight) / radius
                return -77.93 - radius * math.sin(theta), 2.0 * radius - radius * (1.0 - math.cos(theta))
            return -77.0 + distance - start - 2.0 * half_curve - straight, 0.0

        return [point_at(i * self.cfg.point_gap) for i in range(int(self.cfg.total_distance / self.cfg.point_gap) + 1)]

    def _resample_command(self, env_ids: Sequence[int]):
        self.next_target_idx[env_ids] = 1
        self.completed[env_ids] = False
        self.just_reached[env_ids] = False
        self.vel_command_b[env_ids] = 0.0

    def _update_metrics(self):
        return None

    def _update_command(self):
        position_xy = self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
        target_idx = self.next_target_idx.clamp(max=len(self.track_points) - 1)
        target_xy = self.track_points[target_idx]
        delta_xy = target_xy - position_xy
        distance = torch.linalg.vector_norm(delta_xy, dim=-1)

        previous_xy = self.track_points[(target_idx - 1).clamp_min(0)]
        segment_xy = target_xy - previous_xy
        segment_length = torch.linalg.vector_norm(segment_xy, dim=-1).clamp_min(1.0e-6)
        passed_distance = ((position_xy - target_xy) * segment_xy).sum(dim=-1) / segment_length
        lateral_error = torch.abs((position_xy - target_xy)[:, 0] * segment_xy[:, 1] - (position_xy - target_xy)[:, 1] * segment_xy[:, 0]) / segment_length
        reached = (~self.completed) & ((distance < self.cfg.reach_threshold) | ((passed_distance > 0.0) & (lateral_error < self.cfg.pass_lateral_threshold)))
        self.just_reached = reached
        self.metrics["waypoints_reached"] += reached.float()
        self.next_target_idx = torch.where(reached, self.next_target_idx + 1, self.next_target_idx)
        self.completed |= self.next_target_idx >= len(self.track_points)

        target_idx = self.next_target_idx.clamp(max=len(self.track_points) - 1)
        delta_xy = self.track_points[target_idx] - position_xy
        delta_world = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        delta_world[:, :2] = delta_xy
        delta_body = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), delta_world)
        bearing = torch.atan2(delta_body[:, 1], delta_body[:, 0])
        speed = torch.linalg.vector_norm(delta_body[:, :2], dim=-1).mul(self.cfg.speed_gain)
        speed = torch.clamp(speed, self.cfg.min_speed, self.cfg.max_speed)
        # Stage 1 deliberately trains only forward speed and yaw rate.  Keep
        # navigation in that same command space instead of introducing lateral
        # sidestepping only when Stage 2 begins.
        self.vel_command_b[:, 0] = speed * torch.cos(bearing).clamp_min(0.0)
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = torch.clamp(self.cfg.yaw_gain * wrap_to_pi(bearing), -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        self.vel_command_b[self.completed] = 0.0


@configclass
class WaypointVelocityCommandCfg(CommandTermCfg):
    """Turn a fixed ordered ground-plane path into base-frame velocity commands."""

    class_type: type = WaypointVelocityCommand
    asset_name: str = "robot"
    point_gap: float = 2.0
    total_distance: float = 400.0
    reach_threshold: float = 0.9
    pass_lateral_threshold: float = 1.25
    speed_gain: float = 0.8
    min_speed: float = 0.25
    max_speed: float = 1.2
    yaw_gain: float = 1.5
    max_yaw_rate: float = 1.0
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
