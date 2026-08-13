"""Flat-ground G1 walking task derived from Isaac Lab's official G1 task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from .mdp import WaypointVelocityCommandCfg, completed, completion_bonus, failed_termination, reached_waypoint


@configclass
class G1VelocityWalkFlatEnvCfg(G1FlatEnvCfg):
    """Train a reproducible forward-walking gait before waypoint navigation."""

    def __post_init__(self):
        super().__post_init__()
        # This is deliberately per GPU process.  With two 72 GB GPUs the
        # default torchrun command therefore simulates 12,288 G1 instances.
        # The large vectorized batch keeps PhysX and PPO on the GPUs instead
        # of spending most wall time in Python/CPU scheduling.
        self.scene.num_envs = 6144
        self.scene.env_spacing = 2.0
        self.episode_length_s = 20.0

        # First-stage curriculum: stable forward walking only.  Navigation is
        # deliberately omitted and will be added after this gait is validated.
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # The course has curves, so yaw tracking belongs in the gait stage.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # The flat task does not need cameras or height scanning.  Keep the
        # official proprioception/contact-based locomotion design intact.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        # GPU PhysX capacity for 6144 articulated G1 instances per process.
        # No camera, ray-caster, or CPU-side waypoint calculations are used.
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_max_num_partitions = 32


@configclass
class G1WaypointRaceEnvCfg(G1VelocityWalkFlatEnvCfg):
    """Official G1 gait task driven through the ordered 400 m target points."""

    def __post_init__(self):
        super().__post_init__()
        # 400 m at the capped command speed needs more than 333 seconds.
        self.episode_length_s = 480.0
        self.commands.base_velocity = WaypointVelocityCommandCfg(asset_name="robot")
        self.rewards.reached_waypoint = RewTerm(func=reached_waypoint, weight=2.0)
        self.rewards.completion_bonus = RewTerm(func=completion_bonus, weight=50.0)
        self.rewards.termination_penalty.func = failed_termination
        self.terminations.completed = DoneTerm(func=completed)
