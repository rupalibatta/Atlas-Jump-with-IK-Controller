"""Atlas jumping trajectory generator for Caltech ME 133a.

Reconstructed from Appendix 8 of the December 2024 final report.
Requires the course's GeneratorNode / TransformHelpers and KinematicChain.
The report contains simulation code, not a dynamics-based physical robot controller.
"""
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from hw5code.GeneratorNode import GeneratorNode
from hw5code.TransformHelpers import T_from_Rp, pxyz, Rotz, Transform_from_T
from KinematicChain import KinematicChain


class Trajectory:
    def __init__(self, node):
        self.node = node
        self.broadcaster = TransformBroadcaster(node)
        self.chain_left = KinematicChain(node, 'pelvis', 'l_foot', self.joint_names_left())
        self.chain_right = KinematicChain(node, 'pelvis', 'r_foot', self.joint_names_right())
        self.squat_duration = 3.0
        self.rise_duration = 1.5
        self.gravity = -9.81
        self.spin_rate = np.pi
        self.rotation_angle = 0.0
        self.joint_names = self.joint_names_all()
        self.qd = np.zeros(len(self.joint_names))
        self.qdot = np.zeros(len(self.joint_names))
        self.qinit_left = np.zeros(len(self.joint_names_left()))
        self.qinit_right = np.zeros(len(self.joint_names_right()))
        self.reset_to_standing_position()
        self.state = 'squat down'
        self.timer = 0.0
        self.vertical_velocity = 0.0
        self.pelvis_height = 1.0

    def jointnames(self):
        return self.joint_names_all()

    @staticmethod
    def joint_names_all():
        return [
            'back_bkx', 'back_bky', 'back_bkz',
            'l_arm_elx', 'l_arm_ely', 'l_arm_shx', 'l_arm_shz',
            'l_arm_wrx', 'l_arm_wry', 'l_arm_wry2',
            'l_leg_akx', 'l_leg_aky', 'l_leg_hpx', 'l_leg_hpy',
            'l_leg_hpz', 'l_leg_kny', 'neck_ry',
            'r_arm_elx', 'r_arm_ely', 'r_arm_shx', 'r_arm_shz',
            'r_arm_wrx', 'r_arm_wry', 'r_arm_wry2',
            'r_leg_akx', 'r_leg_aky', 'r_leg_hpx', 'r_leg_hpy',
            'r_leg_hpz', 'r_leg_kny',
        ]

    @staticmethod
    def joint_names_left():
        return ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy',
                'l_leg_kny', 'l_leg_aky', 'l_leg_akx']

    @staticmethod
    def joint_names_right():
        return ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy',
                'r_leg_kny', 'r_leg_aky', 'r_leg_akx']

    def reset_to_standing_position(self):
        self.qd = np.zeros(len(self.joint_names))
        self.qdot = np.zeros(len(self.joint_names))
        self.pelvis_height = 1.0

    def evaluate(self, t, dt):
        # Squat -> rise -> ascent -> descent -> landing.
        if self.state == 'squat down':
            self.timer += dt
            self.pelvis_height = 1.0 - 0.5 * (self.timer / self.squat_duration)
            if self.timer >= self.squat_duration:
                self.state, self.timer = 'rise up', 0.0
        elif self.state == 'rise up':
            self.timer += dt
            self.pelvis_height = 0.5 + 0.25 * (self.timer / self.rise_duration)
            if self.timer >= self.rise_duration:
                self.state, self.timer = 'jump', 0.0
                self.vertical_velocity = 5.0
        elif self.state == 'jump':
            self.vertical_velocity += self.gravity * dt
            self.pelvis_height += self.vertical_velocity * dt
            self.rotation_angle += self.spin_rate * dt
            if self.vertical_velocity <= 0:
                self.state = 'flight'
        elif self.state == 'flight':
            self.vertical_velocity += self.gravity * dt
            self.pelvis_height += self.vertical_velocity * dt
            self.rotation_angle += self.spin_rate * dt
            if self.pelvis_height <= 1.0:
                self.pelvis_height = 1.0
                self.state = 'land'
        elif self.state == 'land':
            self.vertical_velocity = 0.0
            self.state, self.timer = 'squat down', 0.0

        p_pelvis = pxyz(0.0, 0.0, self.pelvis_height)
        T_pelvis = T_from_Rp(Rotz(self.rotation_angle), p_pelvis)
        trans = TransformStamped()
        trans.header.stamp = self.node.get_clock().now().to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id = 'pelvis'
        trans.transform = Transform_from_T(T_pelvis)
        self.broadcaster.sendTransform(trans)

        # Adjustable world-frame stance; the original appendix uses +/-0.5 m.
        left_foot_world = np.eye(4)
        left_foot_world[1, 3] = 0.5
        left_foot_world[2, 3] = 0.0  # Set to 0.1 for the raised-block experiment.
        right_foot_world = np.eye(4)
        right_foot_world[1, 3] = -0.5
        right_foot_world[2, 3] = 0.0

        inverse_pelvis = np.linalg.inv(T_pelvis)
        left_target = inverse_pelvis @ left_foot_world
        right_target = inverse_pelvis @ right_foot_world
        left_angles, left_ok = self.chain_left.inverse_kinematics(
            left_target, qinit=self.qinit_left)
        if left_ok:
            self.qinit_left = left_angles
            for name, angle in zip(self.joint_names_left(), left_angles):
                self.qd[self.joint_names.index(name)] = angle
        right_angles, right_ok = self.chain_right.inverse_kinematics(
            right_target, qinit=self.qinit_right)
        if right_ok:
            self.qinit_right = right_angles
            for name, angle in zip(self.joint_names_right(), right_angles):
                self.qd[self.joint_names.index(name)] = angle
        for name in set(self.joint_names) - set(self.joint_names_left() + self.joint_names_right()):
            self.qd[self.joint_names.index(name)] = 0.0
        # The source constructs a JointState but leaves its publish call commented out.
        return self.qd, self.qdot


def main(args=None):
    rclpy.init(args=args)
    node = GeneratorNode('jump_generator', 100, Trajectory)
    node.spin()
    node.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
