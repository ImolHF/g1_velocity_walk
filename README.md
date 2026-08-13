# G1 Velocity Walk

Independent two-stage project for a Unitree G1 400 m run.  It reuses Isaac
Lab's official flat-ground G1 velocity-locomotion task for gait learning, then
uses the same policy and locomotion rewards to follow the ordered track points.

## What the waypoint task does

The track contains 201 coordinates: the start at 0 m plus one point every 2 m
through 400 m.  The robot has no camera, LiDAR, or map perception.  At every
simulation step, the environment uses the robot's simulator state to calculate
the direction from the robot to its next point and supplies that as the usual
base-velocity command (forward speed plus yaw rate).  The existing official G1
policy already observes this command and receives its normal
gait/velocity-tracking rewards.  Entering a
0.9 m circle around a point (or passing it closely) advances only that
environment to the next point; the episode ends after the final point.  A
finish is rewarded, while the original fall penalty is retained only for
actual failures.

## Server installation

The server must have the same Isaac Lab / Isaac Sim version as this project
and access to the official Isaac Lab G1 asset (via the Isaac Sim asset cache or
Nucleus).  This project intentionally has no path to the local Windows
`unitree_model` directory.

```bash
cd g1_velocity_walk
conda activate isaaclab
python -m pip install -e source/g1_velocity_walk
```

## Stage 1: train walking

```bash
python -m pip install -e source/g1_velocity_walk
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \\
  --task G1VelocityWalk-Flat-v0 --headless --distributed --num_envs 4096 \\
  --max_iterations 3000 --run_name walk_stage1
```

`--num_envs 4096` is per GPU, so this uses 8192 parallel environments across
two cards.  First run a small smoke test by replacing it with `--num_envs 256
--max_iterations 2`.  If your server has CPU headroom, increase the per-GPU
count gradually; do not change the command to a total count.

## Stage 2: ordered 400 m waypoints

Find the final Stage 1 checkpoint under
`logs/rsl_rl/g1_velocity_walk/<walk-run>/model_*.pt`, then continue from it:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \\
  --task G1WaypointRace400m-v0 --headless --distributed --num_envs 4096 \\
  --max_iterations 5000 --run_name waypoint_stage2 \\
  --resume --load_run '<walk-run>' --checkpoint 'model_.*.pt'
```

For example, if the Stage 1 directory is
`2026-08-13_20-00-00_walk_stage1`, replace `<walk-run>` with that exact
directory name.  The resume arguments now load the selected checkpoint before
Stage 2 starts.
