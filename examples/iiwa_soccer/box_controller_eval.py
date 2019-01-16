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

    # Evaluates the ability of the controller to deal with sliding contact
    # between the ball and the robot. In order of preference, we want the robot
    # to (1) regulate the desired ball deformation while reducing the slip
    # velocity, (2) maintain contact with the ball while reducing the slip
    # velocity, and (3) maintain contact with the ball.
    def EvaluateSlipPerturbation(self, t):
        # Set the states.

        # Perturb the slip velocity.

        # Simulate the system forward by one controller time cycle.

        # Evaluate how well the controller performed.
