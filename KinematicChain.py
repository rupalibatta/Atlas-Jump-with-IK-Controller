"""URDF kinematic chain and damped, null-space IK.

Reconstructed from Appendix 8 of the ME 133a final report (December 2024).
The PDF's character spacing/indentation was corrupted in export. This file
restores syntax and keeps the report's FK, Jacobian, and IK algorithms.
Requires the Caltech course's hw5code.TransformHelpers module.
"""
import enum
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from urdf_parser_py.urdf import Robot
from hw5code.TransformHelpers import (
    T_from_URDF_origin, n_from_URDF_axis, T_from_Rp, Rotn,
    pzero, Reye, p_from_T, R_from_T, cross,
)


class JointType(enum.Enum):
    FIXED = 0
    REVOLUTE = 1
    LINEAR = 2


class URDFStep:
    def __init__(self, name, type, Tshift, nlocal):
        self.name = name
        self.type = type
        self.Tshift = Tshift
        self.nlocal = nlocal
        self.dof = None


class KinematicChain:
    def info(self, message):
        self.node.get_logger().info('KinematicChain: ' + message)

    def error(self, message):
        self.node.get_logger().error('KinematicChain: ' + message)
        raise RuntimeError(message)

    def __init__(self, node, baseframe, tipframe, expectedjointnames):
        self.node = node
        self.joint_names = list(expectedjointnames)
        self.steps = []
        self.dofs = 0
        self.urdf = None
        self.info('Waiting for the URDF to be published...')

        def cb(msg):
            self.urdf = msg.data

        qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        sub = node.create_subscription(String, '/robot_description', cb, qos)
        while self.urdf is None:
            rclpy.spin_once(node)
        node.destroy_subscription(sub)
        robot = Robot.from_xml_string(self.urdf)
        self.info("Processing URDF for robot '%s'" % robot.name)

        frame = tipframe
        while frame != baseframe:
            joint = next((j for j in robot.joints if j.child == frame), None)
            if joint is None:
                self.error("Unable to find joint connecting to '%s'" % frame)
            if joint.parent == frame:
                self.error("Joint '%s' connects frame to itself" % joint.name)
            frame = joint.parent
            if joint.type in ('revolute', 'continuous'):
                kind = JointType.REVOLUTE
            elif joint.type == 'prismatic':
                kind = JointType.LINEAR
            elif joint.type == 'fixed':
                kind = JointType.FIXED
            else:
                self.error("Unknown joint type '%s'" % joint.type)
            nlocal = None
            if kind != JointType.FIXED:
                nlocal = n_from_URDF_axis(joint.axis)
                nlocal = nlocal / np.linalg.norm(nlocal)
            self.steps.insert(0, URDFStep(joint.name, kind,
                                          T_from_URDF_origin(joint.origin), nlocal))

        for step in self.steps:
            if step.type != JointType.FIXED:
                step.dof = self.dofs
                self.dofs += 1
        actual = [s.name for s in self.steps if s.type != JointType.FIXED]
        if actual != self.joint_names:
            self.error('Chain does not match expected joint names: %s' % self.joint_names)

    def fkin(self, q):
        """Return (tip position, tip rotation, linear Jacobian, angular Jacobian)."""
        if len(q) != self.dofs:
            self.error('Number of joint angles does not match chain')
        kinds, positions, axes = [], [], []
        T = np.eye(4)
        for step in self.steps:
            if step.type == JointType.REVOLUTE:
                T = T @ step.Tshift @ T_from_Rp(Rotn(step.nlocal, q[step.dof]), pzero())
            elif step.type == JointType.LINEAR:
                T = T @ step.Tshift @ T_from_Rp(Reye(), step.nlocal * q[step.dof])
            else:
                T = T @ step.Tshift
            if step.type != JointType.FIXED:
                kinds.append(step.type)
                positions.append(p_from_T(T))
                axes.append(R_from_T(T) @ step.nlocal)
        ptip, Rtip = p_from_T(T), R_from_T(T)
        Jv = np.zeros((3, self.dofs))
        Jw = np.zeros((3, self.dofs))
        for i in range(self.dofs):
            if kinds[i] == JointType.REVOLUTE:
                Jv[:, i] = cross(axes[i], ptip - positions[i])
                Jw[:, i] = axes[i]
            elif kinds[i] == JointType.LINEAR:
                Jv[:, i] = axes[i]
        return ptip, Rtip, Jv, Jw

    def inverse_kinematics(self, target_T, qinit=None, max_iters=100, tol=1e-3):
        """Damped Newton-Raphson IK with a null-space knee-posture target."""
        q = np.zeros(self.dofs) if qinit is None else np.copy(qinit)
        for _ in range(max_iters):
            ptip, Rtip, Jv, Jw = self.fkin(q)
            dp = target_T[:3, 3] - ptip
            Rtarget = target_T[:3, :3]
            orientation_error = sum(
                (np.cross(Rtip[:, j], Rtarget[:, j]) for j in range(3)),
                np.zeros(3),
            ) / 2
            error = np.hstack((dp, orientation_error))
            if np.linalg.norm(error) < tol:
                return q, True
            J = np.vstack((Jv, Jw))
            gamma = 0.05
            J_pinv = J.T @ np.linalg.pinv(J @ J.T + gamma**2 * np.eye(6))
            dq_primary = J_pinv @ error
            q_target = np.copy(q)
            # Each six-joint leg chain contains one knee at local index 3.
            # The original appendix checked both names, despite solving each leg separately.
            for knee in ('l_leg_kny', 'r_leg_kny'):
                if knee in self.joint_names:
                    q_target[self.joint_names.index(knee)] = np.radians(45)
            secondary_velocity = -0.10 * (q - q_target)
            nullspace = np.eye(self.dofs) - J_pinv @ J
            q += dq_primary + nullspace @ secondary_velocity
        return q, False
