# TODO: turn this system into an actual discrete system (estimated time required: 30m)
import math
import numpy as np
import logging
from manipulation_plan import ManipulationPlan
from embedded_box_soccer_sim import EmbeddedSim
from controller_base import ControllerBase

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType, BasicVector, MultibodyForces,
        SpatialVelocity, ComputePoseDiffInCommonFrame)
from pydrake.solvers import mathematicalprogram

# All code specific to controlling the box.
class BoxController(ControllerBase):
    def __init__(self, sim_dt, penetration_allowance, robot_type, all_plant, robot_plant, mbw, robot_instance,
            ball_instance, fully_actuated=False):
        ControllerBase.__init__(self, sim_dt, penetration_allowance, robot_type, all_plant, robot_plant, mbw, robot_instance, ball_instance)

        # Save the robot type.
        self.set_name('box_controller')

        # Construct the plan.
        self.plan = ManipulationPlan()

        # Save the robot and ball instances.
        self.robot_instance = robot_instance
        self.ball_instance = ball_instance

        # Get the plants.
        self.robot_plant = robot_plant
        self.robot_and_ball_plant = all_plant
        self.mbw = mbw

        # Save whether the plant is fully actuated.
        self.fully_actuated = fully_actuated

        # Set the output size.
        self.command_output_size = self.robot_and_ball_plant.num_velocities()

        # Set PID gains.
        self.cartesian_kp = np.ones([3, 1]) * 60
        self.cartesian_kd = np.ones([3, 1]) * 30

        # Joint gains for the robot.
        self.num_robot_actuators()
        self.robot_gv_kp = np.ones([self.nv_robot()]) * 10
        self.robot_gv_ki = np.ones([self.nv_robot()]) * 0.1
        self.robot_gv_kd = np.ones([self.nv_robot()]) * 1.0

        # Actuator limits.
        self.actuator_limit = float('inf')

        # TODO: delete the line below.
        logging.warning('box_controller.py is violating the const System assumption')
        self.ball_accel_from_controller = np.array([0, 0, 0])

    # NOTE: Not sure why it's necessary to call the base class for BoxController.FindBallGroundContacts() to work.
    def FindBallGroundContacts(self, all_q):
        return ControllerBase.FindRobotBallContacts(self, all_q)

    # Gets the foot links from the robot and ball plant.
    def get_foot_links_from_robot_and_ball_plant(self):
        return [ self.robot_and_ball_plant.GetBodyByName("box") ]

    # Gets the foot links from the robot tree.
    def get_foot_links_from_robot_plant(self):
        return [ self.robot_plant.GetBodyByName("box") ]

    # Computes the control forces when contact is not desired. This scheme simply uses inverse dynamics-based control to
    # follow the desired plan without accounting for any contact forces. The idea being that the plan was formulated to
    # pass over the ball, so that any contact is (hopefully) transient.
    def ComputeActuationForContactNotDesired(self, context):
        # Get the desired robot acceleration.
        q_robot_des = self.plan.GetRobotQVAndVdot(
                context.get_time())[0:nv_robot()-1]
        qdot_robot_des = self.plan.GetRobotQVAndVdot(
                context.get_time())[nv_robot(), 2*nv_robot()-1]
        qddot_robot_des = self.plan.GetRobotQVAndVdot(
                context.get_time())[-nv_robot():]

        # Get the robot current generalized position and velocity.
        q_robot = get_q_robot(context)
        qd_robot = get_v_robot(context)

        # Set qddot_robot_des using error feedback.
        qddot = qddot_robot_des + np.diag(self.robot_gv_kp).dot(q_robot_des - q_robot) + np.diag(self.robot_gv_ki).diag(get_integral_value(context)) + np.diag(self.robot_gv_kd).dot(qdot_robot_des - qd_robot)

        # Set the state in the robot context to q_robot and qd_robot.
        x = self.robot_plant.tree().get_mutable_multibody_state_vector(robot_context)
        assert len(x) == len(q_robot) + len(qd_robot)
        x[0:len(q_robot)-1] = q_robot
        x[-len(qd_robot):] = qd_robot

        # Get the generalized inertia matrix.
        M = self.robot_plant.tree().CalcMassMatrixViaInverseDynamics(robot_context)

        # Compute the contribution from force elements.
        robot_tree = self.robot_plant.tree()
        link_wrenches = MultibodyForces(robot_tree)

        # Compute the external forces.
        fext = -robot_tree.CalcInverseDynamics(robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

        # Compute inverse dynamics.
        u = M * qddot - fext
        return u

    def ComputeContributionsFromBallTrackingErrors(self, q0, v0, normal_and_signed_distance_data):
        normal_foot_ball_W = normal_and_signed_distance_data.normal_foot_ball_W
        phi = normal_and_signed_distance_data.phi
        closest_foot_body = normal_and_signed_distance_data.closest_foot_body

        all_plant = self.robot_and_ball_plant
        robot_plant = self.robot_plant

        # The closest foot body was defined in the "all plant". Find the corresponding body in the robot plant.
        closest_foot_body = robot_plant.GetBodyByName(closest_foot_body.name())

        # Using RMRC, convert the normal vector and signed distance to a change in positions in generalized velocity
        # coordinates.
        phi_des = self.get_desired_ball_signed_distance_under_contact()
        robot_plant.SetPositions(self.robot_context, all_plant.GetPositionsFromArray(self.robot_instance, q0))
        spatial_v = np.zeros([6])
        spatial_v[3:6] = -normal_foot_ball_W * (phi_des - phi)
        v_from_ball_position_tracking = self.RMRC(robot_plant, self.robot_context,
                spatial_v, closest_foot_body.body_frame())

        # Get the ball velocity in the direction of the normal and use RMRC to match it.
        all_plant.SetVelocities(self.robot_and_ball_context, v0)
        ball_translational_velocity = all_plant.EvalBodySpatialVelocityInWorld(
                self.robot_and_ball_context, self.get_ball_from_robot_and_ball_plant()).translational()
        ball_translational_velocity_normal_W = normal_foot_ball_W * np.inner(
                normal_foot_ball_W, ball_translational_velocity)
        robot_plant.SetPositions(self.robot_context, all_plant.GetPositionsFromArray(self.robot_instance, q0))
        spatial_v = np.zeros([6])
        spatial_v[3:6] = ball_translational_velocity_normal_W
        v_from_ball_velocity_tracking = self.RMRC(robot_plant, self.robot_context,
                spatial_v, closest_foot_body.body_frame())

        return [ v_from_ball_position_tracking, v_from_ball_velocity_tracking ]

    def ComputeContributionsFromPlannedBoxVelocity(self, q0, v0, normal_and_signed_distance_data,
            q_robot_planned, v_robot_planned):
        normal_foot_ball_W = normal_and_signed_distance_data.normal_foot_ball_W
        phi = normal_and_signed_distance_data.phi
        closest_foot_body = normal_and_signed_distance_data.closest_foot_body

        all_plant = self.robot_and_ball_plant
        robot_plant = self.robot_plant

        # The closest foot body was defined in the "all plant". Find the corresponding body in the robot plant.
        closest_foot_body = robot_plant.GetBodyByName(closest_foot_body.name())

        # Get the desired spatial velocity for the end effector, first removing components along the normal direction.
        # TODO: We're using q0 below. Use q_robot_planned instead?
        robot_plant.SetPositions(self.robot_context, all_plant.GetPositionsFromArray(self.robot_instance, q0))
        robot_plant.SetVelocities(self.robot_context, v_robot_planned)
        V_WE_des_spatial = robot_plant.EvalBodySpatialVelocityInWorld(self.robot_context, closest_foot_body)
        V_WE_des = np.zeros([6])
        V_WE_des[0:3] = V_WE_des_spatial.rotational()
        V_WE_des[3:6] = V_WE_des_spatial.translational()
        V_WE_des[3:6] -= normal_foot_ball_W * np.inner(V_WE_des[3:6], normal_foot_ball_W)

        # Using RMRC, convert the altered end effector velocity to a desired velocity in generalized velocity coordinates.
        # NOTE: We are computing RMRC from the *current* robot configuration, not the planned one.
        robot_plant.SetVelocities(self.robot_context, np.zeros([robot_plant.num_velocities()]))
        robot_plant.SetPositions(self.robot_context, all_plant.GetPositionsFromArray(self.robot_instance, q0))
        v_from_planned_velocity = self.RMRC(robot_plant, self.robot_context, V_WE_des, closest_foot_body.body_frame())
        return v_from_planned_velocity

    def ComputeContributionsFromPlannedBoxPosition(self, q0, normal_and_signed_distance_data, q_robot_planned):
        normal_foot_ball_W = normal_and_signed_distance_data.normal_foot_ball_W
        phi = normal_and_signed_distance_data.phi
        closest_foot_body = normal_and_signed_distance_data.closest_foot_body

        all_plant = self.robot_and_ball_plant
        robot_plant = self.robot_plant

        # Get the spatial pose differential between the actual and planned pose of the end effector in spatial velocity
        # coordinates. X_WE represents the transformation from the end-effector frame to the world frame.
        all_plant.SetPositions(self.robot_and_ball_context, q0)
        X_WE_robot = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, closest_foot_body)
        all_plant.SetPositions(self.robot_and_ball_context, self.robot_instance, q_robot_planned)
        X_WE_robot_des = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, closest_foot_body)

        # Convert the difference in poses to a differential in spatial velocity coordinates (expressed in world frame).
        dX_WE = ComputePoseDiffInCommonFrame(X_WE_robot, X_WE_robot_des)

        # Remove the translational components in the direction of the normal vector from this spatial pose differential.
        spatial_normal_W = np.zeros([6])
        spatial_normal_W[-3:] = normal_foot_ball_W
        dX_WE -= spatial_normal_W * np.inner(dX_WE, spatial_normal_W)

        # The closest foot body was defined in the "all plant". Find the corresponding body in the robot plant.
        closest_foot_body = robot_plant.GetBodyByName(closest_foot_body.name())

        # Using RMRC, convert the altered difference between the actual and planned end effector configurations to
        # a positional change in generalized velocity coordinates.
        robot_plant.SetVelocities(self.robot_context, np.zeros([robot_plant.num_velocities()]))
        robot_plant.SetPositions(self.robot_context, all_plant.GetPositionsFromArray(self.robot_instance, q0))
        v_from_planned_position = self.RMRC(robot_plant, self.robot_context, dX_WE, closest_foot_body.body_frame())
        return v_from_planned_position

    '''
    Computes the desired positions and velocities for the robot to satisfy the amount of interpenetration (a proxy for
    deformation) between the robot and the ball *while also attempting to follow the robot's initially desired
    trajectory*.

    This function first computes the error in R3 between the desired signed distance between the robot end
    effector/ball and the actual signed distance, and same for the time derivatives (we assume that the desired
    time derivative of the signed distance is zero). Then, a spatial differential between the planned and actual
    pose of the box is computed, as well as the time derivative of this differential. The components of the
    spatial differential/time derivative parallel to the error/time derivative are then removed. The error/time
    derivative and the spatial differential/time derivative are then summed. RMRC turns the resulting spatial
    differential into a change in generalized coordinates, and the resulting time derivative of the spatial
    differential into a change in generalized velocities. These deltas (changes) are then scaled by some gains and
    summed with the generalized acceleration planned for the robot. The summed generalized acceleration is then
    transformed using inverse dynamics into a control force.
    '''
    def ComputeActuationForContactDesired(self, controller_context, q0, v0, q_robot_planned, v_robot_planned,
             vdot_robot_planned):

        # Get the current time.
        t = controller_context.get_time()

        # Get the relevant plants.
        all_plant = self.robot_and_ball_plant
        robot_plant = self.robot_plant

        # Get the signed distance and the normal vector between the witness points on the robot end effector and the
        # ball.
        normal_and_signed_distance_data = self.GetNormalAndSignedDistanceFromRobotToBall(q0)

        # Compute contributions from error between the end effector location/velocity and the ball location/velocity.
        v_from_ball_position_tracking, v_from_ball_velocity_tracking = self.ComputeContributionsFromBallTrackingErrors(
                q0, v0, normal_and_signed_distance_data)

        # Compute contributions from the error in the planned position for the robot.
        v_from_planned_position = self.ComputeContributionsFromPlannedBoxPosition(q0,
                normal_and_signed_distance_data, q_robot_planned)

        # Compute contributions from the error in the planned position for the robot.
        v_from_planned_velocity = self.ComputeContributionsFromPlannedBoxVelocity(q0, v0,
                normal_and_signed_distance_data, q_robot_planned, v_robot_planned)

        # Turn the feedback into accelerations.
        vdot = np.reshape(vdot_robot_planned, [-1, 1]) + \
                np.diag(self.robot_gv_kp).dot(v_from_planned_position + v_from_ball_position_tracking) + \
                np.diag(self.robot_gv_kd).dot(v_from_planned_velocity + v_from_ball_velocity_tracking)

        # Get the generalized inertia matrix.
        robot_plant.SetPositions(self.robot_context, all_plant.GetPositionsFromArray(self.robot_instance, q0))
        M = robot_plant.CalcMassMatrixViaInverseDynamics(self.robot_context)

        # Compute the contribution from force elements.
        link_wrenches = MultibodyForces(robot_plant)

        # Compute the external forces.
        fext = np.reshape(-robot_plant.CalcInverseDynamics(
            self.robot_context, np.zeros([self.nv_robot()]), link_wrenches), (-1, 1))

        # Compute inverse dynamics.
        return M.dot(vdot) - fext

    # Loads all plans into the controller.
    def LoadPlans(self, path):
        from pydrake.common import FindResourceOrThrow

        # Set the Drake prefix.
        prefix = 'drake/examples/iiwa_soccer/'

        # Add a '/' to the end of the path if necessary.
        if path[-1] != '/':
            path += '/'

        # TODO: move the iiwa plans below into the forthcoming IiwaController.
        '''
        # Read in the plans for the robot.
        if self.robot_type == 'iiwa':
            self.plan.ReadIiwaRobotQVAndVdot(
                    FindResourceOrThrow(prefix + 'plan/joint_timings_fit.mat'),
                    FindResourceOrThrow(prefix + 'plan/joint_angle_fit.mat'),
                    FindResourceOrThrow(prefix + 'plan/joint_vel_fit.mat'),
                    FindResourceOrThrow(prefix + 'plan/joint_accel_fit.mat'))

            # Read in the plans for the point of contact.
            self.plan.ReadContactPoint(
                    FindResourceOrThrow(prefix + 'plan/contact_pt_timings.mat'),
                    FindResourceOrThrow(prefix + 'plan/contact_pt_positions.mat'),
                    FindResourceOrThrow(prefix + 'plan/contact_pt_velocities.mat'))

            # Read in the plans for the ball kinematics.
            self.plan.ReadBallQVAndVdot(
                    FindResourceOrThrow(prefix + 'plan/ball_timings.mat'),
                    FindResourceOrThrow(prefix + 'plan/ball_com_positions.mat'),
                    FindResourceOrThrow(prefix + 'plan/ball_quats.mat'),
                    FindResourceOrThrow(prefix + 'plan/ball_com_velocities.mat'),
                    FindResourceOrThrow(prefix + 'plan/ball_omegas.mat'),
                    FindResourceOrThrow(prefix + 'plan/ball_com_accelerations.mat'),
                    FindResourceOrThrow(prefix + 'plan/ball_alphas.mat'),
                    FindResourceOrThrow(prefix + 'plan/contact_status.mat'))
        '''

        self.plan.ReadBoxRobotQVAndVdot(
                    FindResourceOrThrow(prefix + path + 'timings.mat'),
                    FindResourceOrThrow(prefix + path + 'box_positions.mat'),
                    FindResourceOrThrow(prefix + path + 'box_quats.mat'),
                    FindResourceOrThrow(prefix + path + 'box_linear_vel.mat'),
                    FindResourceOrThrow(prefix + path + 'box_angular_vel.mat'),
                    FindResourceOrThrow(prefix + path + 'box_linear_accel.mat'),
                    FindResourceOrThrow(prefix + path + 'box_angular_accel.mat'))

        # Read in the plans for the point of contact.
        self.plan.ReadContactPoint(
                    FindResourceOrThrow(prefix + path + 'timings.mat'),
                    FindResourceOrThrow(prefix + path + 'contact_pt_positions.mat'),
                    FindResourceOrThrow(prefix + path + 'contact_pt_velocities.mat'))

        # Read in the plans for the ball kinematics.
        self.plan.ReadBallQVAndVdot(
                    FindResourceOrThrow(prefix + path + 'timings.mat'),
                    FindResourceOrThrow(prefix + path + 'ball_com_positions.mat'),
                    FindResourceOrThrow(prefix + path + 'ball_quats.mat'),
                    FindResourceOrThrow(prefix + path + 'ball_com_velocities.mat'),
                    FindResourceOrThrow(prefix + path + 'ball_omegas.mat'),
                    FindResourceOrThrow(prefix + path + 'ball_com_accelerations.mat'),
                    FindResourceOrThrow(prefix + path + 'ball_alphas.mat'),
                    FindResourceOrThrow(prefix + path + 'contact_status.mat'))

    # Calculate what forces to apply to the joints.
    def DoControlCalc(self, context, output):
        # Determine whether we're in a contacting or not-contacting phase.
        contact_desired = self.plan.IsContactDesired(context.get_time())

        # Get the generalized positions.
        q = self.get_q_all(context)

        # Look for full actuation.
        if self.fully_actuated:
            tau = self.ComputeFullyActuatedBallControlForces(context)

        else:  # "Real" control.
            # Compute tau.
            if contact_desired == True:
                v = self.get_v_all(context)
                planned_robot_kinematics = self.plan.GetRobotQVAndVdot(context.get_time())
                q_robot_planned = planned_robot_kinematics[0:self.robot_plant.num_positions()]
                v_robot_planned = \
                        planned_robot_kinematics[self.robot_plant.num_positions():-self.robot_plant.num_velocities()]
                vdot_robot_planned = planned_robot_kinematics[-self.robot_plant.num_velocities():]
                print('Computing actuation')
                tau = self.ComputeActuationForContactDesired(
                        context, q, v, q_robot_planned, v_robot_planned, vdot_robot_planned)
                print('Done')
            else:
                # No contact desired.
                logging.info('Contact not desired at time ' + str(context.get_time()))
                tau = self.ComputeActuationForContactNotDesired(context)


        # Set the torque output.
        mutable_torque_out = output.get_mutable_value()
        mutable_torque_out[:] = np.zeros(mutable_torque_out.shape)

        if self.fully_actuated:
            mutable_torque_out[:] = tau
        else:
            self.robot_and_ball_plant.SetVelocitiesInArray(self.robot_instance, tau.flatten(), mutable_torque_out)

    def DoCalcTimeDerivatives(self, context, derivatives):
        # Determine whether we're in a contacting or not-contacting phase.
        contact_intended = self.plan.IsContactDesired(context.get_time())

        if contact_intended:
            derivatives.get_mutable_vector().SetFromVector(np.zeros([self.nq_robot()]))
        else:
            # Get the desired robot configuration.
            q_robot_des = self.plan.GetRobotQVAndVdot(
                    context.get_time())[0:nv_robot()-1]

            # Get the current robot configuration.
            q_robot = get_q_robot(context)
            derivatives.get_mutable_vector().SetFromVector(q_robot_des - q_robot)

