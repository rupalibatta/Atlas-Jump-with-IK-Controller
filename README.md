# Atlas Humanoid Jumping

A ROS 2 simulation of **Atlas humanoid jumping** developed for **Caltech ME 133a**. We used forward and inverse kinematics to generate squats, jumps, and midair rotations from different starting leg positions, including a wide frog stance and one foot elevated on a block.

## Project outline

1. Model Atlas's two six-degree-of-freedom legs using URDF-based kinematic chains.
2. Generate a squat, takeoff, flight, and landing trajectory with an explicit state machine.
3. Solve each leg's joint angles independently to accommodate asymmetric foot positions.
4. Use damped Jacobian inverse kinematics and a secondary knee-posture task to avoid backwards knee bending.
5. Test frog jumps, uneven starting heights, half twists, and ballet-inspired spinning jumps in RViz.

## Jump trajectory and inverse kinematics

The controller tracks a desired pelvis pose in the world frame while specifying foot targets for each leg. A state machine lowers the pelvis for a squat, raises it for takeoff, then integrates upward velocity with simulated gravity during flight. A rotation about the vertical axis produces twisting jumps. Each foot target is transformed into the pelvis frame before its leg's joint angles are found through iterative inverse kinematics.

The kinematic solver computes joint transforms and the linear and angular Jacobians from the robot's URDF. It uses a damped pseudoinverse for pose corrections:

$$
J^\dagger = J^T(JJ^T + \gamma^2 I)^{-1}.
$$

A secondary joint-posture correction is projected through $(I-J^\dagger J)$ to encourage forward knee bending without replacing the primary end-effector target.

## Results

The report demonstrates squat and basic jump motions, a wide-stance frog jump, jumps with one foot initially raised, a half-spin jump, and a second-position pirouette. Joint trajectories from four consecutive basic jumps were reported as smooth and approximately parabolic. These are kinematic animations with prescribed gravity, rather than a full center-of-mass or contact-dynamics simulation.

**[Final demonstration video](https://tinyurl.com/me133jump)**

## Repository contents

| File | Contents |
| --- | --- |
| [`atlas_jump.py`](atlas_jump.py) | ROS 2 trajectory generator, jump state machine, and independent leg IK targets |
| [`KinematicChain.py`](KinematicChain.py) | URDF forward kinematics, Jacobian computation, and damped inverse kinematics |
| [`ME133a_FINAL_REPORT.pdf`](ME133a_FINAL_REPORT.pdf) | Project description, demonstrations, joint plots, and original code appendix |
| [`README.md`](README.md) | Project overview |


## Authors

**Claire Hays, Rupali Batta, and Jana Woo**  
California Institute of Technology, **ME 133a**
