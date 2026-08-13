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

# Avoid two training processes oversubscribing CPU BLAS/OpenMP threads.
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

The environment uses GPU PhysX, Fabric, GPU-resident Torch waypoint math, and
no cameras or height scanner.  CPU is still required to launch two Isaac Sim
processes, so no configuration can guarantee it will never limit throughput;
the thread limits above prevent the usual CPU oversubscription problem.

## Stage 1: train walking

```bash
python -m pip install -e source/g1_velocity_walk
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \\
  --task G1VelocityWalk-Flat-v0 --headless --distributed --num_envs 6144 \\
  --max_iterations 3000 --run_name walk_stage1
```

`--num_envs` is per GPU, not a total.  The recommended run uses 6144 on each
card (12,288 total) and is tuned for two 72 GB GPUs.  First run a smoke test
with `--num_envs 512 --max_iterations 2`, then use 4096 if the server CPU is
still saturated or 8192 only after confirming both GPU memory and simulation
steps/second have headroom.

## Stage 2: ordered 400 m waypoints

Find the final Stage 1 checkpoint under
`logs/rsl_rl/g1_velocity_walk/<walk-run>/model_*.pt`, then continue from it:

```bash
torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \\
  --task G1WaypointRace400m-v0 --headless --distributed --num_envs 6144 \\
  --max_iterations 5000 --run_name waypoint_stage2 \\
  --resume --load_run '<walk-run>' --checkpoint 'model_.*.pt'
```

For example, if the Stage 1 directory is
`2026-08-13_20-00-00_walk_stage1`, replace `<walk-run>` with that exact
directory name.  The resume arguments now load the selected checkpoint before
Stage 2 starts.
