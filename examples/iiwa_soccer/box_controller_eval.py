import math
import numpy as np
import unittest
from manipulation_plan import ManipulationPlan
from box_controller import BoxController
from box_soccer_simulation import BuildBlockDiagram

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType,
                         BasicVector, MultibodyForces, ComputeBasisFromAxis)

class BoxControllerEvaluator:
    def __init__(self):
        self.controller, self.diagram, self.all_plant, self.robot_plant, self.mbw, self.robot_instance, self.ball_instance = BuildBlockDiagram(time_step, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd)

        # Create the context for the diagram.
        self.context = self.diagram.CreateDefaultContext()
        self.controller_context = self.diagram.GetMutableSubsystemContext(self.controller, self.context)
        self.output = self.controller.AllocateOutput()

    # Plugs the planned states for the robot and the ball into the input ports.
    def SetStates(self, t):
        plan = self.controller.plan
        q = np.zeros([self.controller.nq_ball() + self.controller.nq_robot()])
        v = np.zeros([self.controller.nv_ball() + self.controller.nv_robot()])

        # Get the planned q and v.
        q_robot_des = plan.GetRobotQVAndVdot(t)[0:self.controller.nq_robot()]
        v_robot_des = plan.GetRobotQVAndVdot(t)[self.controller.nq_robot():self.controller.nq_robot()+self.controller.nv_robot()]
        q_ball_des = plan.GetBallQVAndVdot(t)[0:self.controller.nq_ball()]
        v_ball_des = plan.GetBallQVAndVdot(t)[self.controller.nq_ball():self.controller.nq_ball()+self.controller.nv_ball()]

        # Set q and v.
        plant = self.controller.robot_and_ball_plant
        plant.SetPositionsInArray(self.robot_instance, q_robot_des, q)
        plant.SetPositionsInArray(self.ball_instance, q_ball_des, q)
        plant.SetVelocitiesInArray(self.robot_instance, v_robot_des, v)
        plant.SetVelocitiesInArray(self.ball_instance, v_ball_des, v)

        # Construct the robot reference positions and velocities and set them equal
        # to the current positions and velocities.
        # Get the robot reference positions, velocities, and accelerations.
        robot_q_input = BasicVector(self.controller.nq_robot())
        robot_v_input = BasicVector(self.controller.nv_robot())
        robot_q_input_vec = robot_q_input.get_mutable_value()
        robot_v_input_vec = robot_v_input.get_mutable_value()
        robot_q_input_vec[:] = q_robot_des.flatten()
        robot_v_input_vec[:] = v_robot_des.flatten()

        # Get the ball reference positions, velocities, and accelerations, and
        # set them equal to the estimated positions and velocities.
        ball_q_input = BasicVector(self.controller.nq_ball())
        ball_v_input = BasicVector(self.controller.nv_ball())
        ball_q_input_vec = ball_q_input.get_mutable_value()
        ball_v_input_vec = ball_v_input.get_mutable_value()
        ball_q_input_vec[:] = q_ball_des.flatten()
        ball_v_input_vec[:] = v_ball_des.flatten()

        # Sanity check.
        for i in range(len(q_robot_des)):
            assert not math.isnan(q_robot_des[i])
        for i in range(len(v_robot_des)):
            assert not math.isnan(v_robot_des[i])

        # Set the robot and ball positions in velocities in the inputs.
        self.controller_context.FixInputPort(
            self.controller.get_input_port_estimated_robot_q().get_index(),
            robot_q_input)
        self.controller_context.FixInputPort(
            self.controller.get_input_port_estimated_robot_v().get_index(),
            robot_v_input)
        self.controller_context.FixInputPort(
            self.controller.get_input_port_estimated_ball_q().get_index(),
            ball_q_input)
        self.controller_context.FixInputPort(
            self.controller.get_input_port_estimated_ball_v().get_index(),
            ball_v_input)

        # Update the context.
        self.context.set_time(t)

        return [q, v]

    # Projects a 2D vector `v` to the (3D) plane defined by the normal `n`.
    def ProjectFrom2DTo3D(v, n):


    # Evaluates the ability of the controller to deal with sliding contact
    # between the ball and the robot. In order of preference, we want the robot
    # to (1) regulate the desired ball deformation while reducing the slip
    # velocity, (2) maintain contact with the ball while reducing the slip
    # velocity, and (3) maintain contact with the ball.
    # Returns a tuple with the first element being:
    #    -1 if the ball is no longer contacting the ground,
    #    -2 if the ball is no longer contacting the robot,
    #     0 otherwise
    # and the second element being the ???
    def EvaluateSlipPerturbation(self, t):
        # Set the states.
        q, v = self.SetStates(t)
        contacts = self.controller.FindContacts()

        # Ensure that the robot/ball and the ball/ground are contacting.
        assert self.controller.IsRobotContactingBall(contacts)
        assert self.controller.IsBallContactingGround(contacts)

        # Verify that there are exactly two points of contact.
        assert len(contacts) == 2

        # Get the amount of interpenetration between the robot and the ball.

        # Get the contact normal.
        n_BA_W = point_pair.nhat_BA_W

        # The slip velocity will be a two-dimensional vector. Sample from
        # a Normal distribution.
        slip_velocity = np.zeros([2])
        slip_velocity[0] = np.random.normal()
        slip_velocity[1] = np.random.normal()

        # Project the 2D vector to the plane defined by the contact normal.
        v_perturb = ProjectFrom2DTo3D(slip_velocity, n_BA_W)

        # Evaluate scene graph's output port, getting a SceneGraph reference.
        all_plant = self.controller.robot_and_ball_plant
        all_context = self.controller.robot_and_ball_context
        query_object = all_plant.EvalAbstractInput(all_context,
            self.controller.geometry_query_input_port.get_index()).get_value()
        inspector = query_object.inspector()

        # Get the ball body.
        ball_body = self.get_ball_from_robot_and_ball_plant()

        # Get the contact and swap bodies (if necessary) so that body_A
        # corresponds to the robot.
        robot_ball_contacts = self.controller.FindRobotBallContacts(q)
        assert len(robot_ball_contacts) == 1
        geometry_A_id = robot_ball_contacts[0].id_A
        geometry_B_id = robot_ball_contacts[0].id_B
        frame_A_id = inspector.GetFrameId(geometry_A_id)
        frame_B_id = inspector.GetFrameId(geometry_B_id)
        body_A = all_plant.GetBodyFromFrameId(frame_A_id)
        body_B = all_plant.GetBodyFromFrameId(frame_B_id)
        if body_B != ball_body:
            # Swap A and B.
            body_A, body_B = body_B, body_A
            robot_ball_contacts[0].p_WCa, robot_ball_contacts[0].p_WCb = robot_ball_contacts[0].p_WCb, robot_ball_contacts[0].p_WCa
            robot_ball_contacts[0].nhat_BA_W *= -1

        # Get the contact point in the robot frame.
        contact_point_W = robot_ball_contacts[0].p_WCa
        X_WA = all_plant.EvalBodyPoseInWorld(all_context, body_A)
        contact_point_A = X_WA.inverse().multiply(contact_point_W)

        # Get the contact Jacobian for the robot.
        J_WAc = self.controller.robot_plant.CalcPointsGeometricJacobianExpressedInWorld(
            self.controller.robot_context, body_B.body_frame(), contact_point_A)

        # Use the Jacobian pseudo-inverse to transform the velocity perturbation
        # to the change in robot generalized velocity.
        v_robot_perturb, residuals, rank, singular_values = np.linalg.lstsq(J_WAc, v_perturb)

        # Simulate the system forward by one controller time cycle.

        # Get the new robot configuration.
        contacts = self.controller.FindContacts(qnew)
    
        # Check whether the ball is still contacting the ground.
        if (not self.controller.IsBallContactingGround(contacts)):
            return [-1, 0]

        # Check whether the ball and robot are still contacting.
        if (not self.controller.IsRobotContactingBall(contacts)):
            return [-2, 0]

        # If still here, all contacts are still maintained. Compute the slip
        # velocities for the ball/ground and ball/robot.

        # Compute the change in slip velocities.
