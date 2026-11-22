# Pick, Navigate, Reach (Team 46)

A simulated Fetch robot picks one of 15 YCB objects off a table, carries it
through a procedurally generated maze, and places it within 10 cm of the green goal sphere. One pipeline
handles every object and every maze; nothing is tuned to a single object.

## Install

```
conda create -n pnr python==3.10
conda activate pnr
pip3 install pybullet numpy scipy matplotlib trimesh
```

## Run one object in one maze (opens the PyBullet window)

```
python pick_nav_reach_test.py <object_idx 0-14> <seed>
python pick_nav_reach_test.py 3 1
```

The script prints the pick attempts, the navigation progress and a final line such as
`RESULT {'success': True, 'stage': 'reach', 'dist_to_goal': 0.0015, 'steps': 13607}`.
A run succeeds when `dist_to_goal` is below 0.1 m. It takes a few minutes in the window because the
environment sleeps 1/240 s per simulation substep.

## Run many objects and mazes without a window

```
python evaluate.py                          # all 15 objects x mazes 0-4
python evaluate.py --seeds 5 6 7 8 9        # other mazes
python evaluate.py --objects 3 11 --seeds 0 1
```

`headless.py` swaps the PyBullet GUI for DIRECT mode and removes the sleep; the physics is unchanged.

## How it works

| Stage | Where | What it does |
|---|---|---|
| Grasp generation | `grasp_planner.py` (`generate_antipodal_grasps`), `pick.py` | Antipodal pairs from the point cloud and normals, gripper opening 10.4 cm, fingers must clear the table, pinch close to the object's centre. Thin objects use relaxed thresholds; flat objects wider than the gripper (the plate) are pinched by the rim. |
| Inverse kinematics | `ik_solver.py` (`solve_arm_ik`) | Damped least squares over the torso and 7 arm joints; the base position is chosen separately. |
| Arm motion planning | `rrt.py` (`plan_arm_path`), `pick.py` (`move_arm`) | RRT-Connect in joint space with collision checks against the table, the object and the robot itself. The straight line is used when it is free. |
| Grasp execution and test | `pick.py` (`pick_object`) | Fold arm, approach 12 cm back, straight approach, close, lift 14 cm. The grasp is then rehearsed (tuck arm, drive, turn, hold 300 steps) and rejected if the object slips or rotates more than 0.05 rad; the simulation state is restored and the next grasp is tried. |
| Navigation | `configuration_space_planner.py` (`plan_configuration_space_path`) | A* over (x, y, heading) with physics collision checks for the robot and a stand-in box for the carried object. |
| Driving and delivery | `pick_nav_reach.py` | Drives the path as slowly as the 15,000-step limit allows, then slides the object onto the goal. |

## Files

- `pick_nav_reach_env.py`: the course environment. The only change is line 1 (`import time`); the original
  `from time import time` makes `time.sleep()` fail when the file is imported.
- `maze_utils.py`, `utils.py`, `keyboard_control.py`, `assets/`: from the course repository, unchanged.

## Notes

- PyBullet runs are not exactly repeatable: a marginal grasp can pass in one run and fail in the next, so
  results should be read over many runs, not one.
- Measured results (15 objects x 5 mazes = 75 runs per set, headless, Linux x86, Python 3.13, PyBullet 3.2.7):
  57/75 on mazes 0-4 (used while developing), 41/75 on mazes 5-9 (used to find a navigation bug) and 50/75 on
  mazes 10-14 (never used for tuning). Maze 9 fails for every object: its goal is in a dead-end pocket the robot
  cannot enter while holding an object (seed 15 behaves the same). The hardest objects are the sugar box, foam brick,
  strawberry, plate and apple. Results differ between machines and runs by a few cases per set.
- Mazes 0-4 were used while developing; `evaluate.py --seeds 10 11 12 13 14` runs mazes that were never used for tuning.
