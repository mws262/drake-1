import argparse
import math
import numpy as np
import unittest
from manipulation_plan import ManipulationPlan
from box_controller import BoxController
from box_soccer_simulation import BuildBlockDiagram

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType,
                         BasicVector, MultibodyForces, ComputeBasisFromAxis,
                         Simulator)

class BoxControllerEvaluator:
    def __init__(self, time_step, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd):
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
    def ProjectFrom2DTo3D(self, v, n):
        R = ComputeBasisFromAxis(axis_index=2, axis_W=n)
        P = R[:,0:2]
        return P.dot(v)

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
    def EvaluateSlipPerturbationAtTime(self, t):
        # Set the states.
        q, v = self.SetStates(t)
        contacts = self.controller.FindContacts(q)

        # Ensure that the robot/ball and the ball/ground are contacting.
        assert self.controller.IsRobotContactingBall(contacts)
        assert self.controller.IsBallContactingGround(contacts)

        # Verify that there are exactly two points of contact.
        assert len(contacts) == 2

        # The slip velocity will be a two-dimensional vector. Sample from
        # a Normal distribution.
        slip_velocity = np.zeros([2,1])
        slip_velocity[0,0] = np.random.normal()
        slip_velocity[1,0] = np.random.normal()

        # Evaluate scene graph's output port, getting a SceneGraph reference.
        all_plant = self.controller.robot_and_ball_plant
        all_context = self.controller.robot_and_ball_context
        query_object = all_plant.EvalAbstractInput(all_context,
            self.controller.geometry_query_input_port.get_index()).get_value()
        inspector = query_object.inspector()

        # Get the ball body.
        ball_body = self.controller.get_ball_from_robot_and_ball_plant()

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

        # Get the amount of interpenetration between the robot and the ball.
        pre_depth = robot_ball_contacts[0].depth

        # Get the contact normal.
        n_BA_W = robot_ball_contacts[0].nhat_BA_W

        # Project the 2D vector to the plane defined by the contact normal.
        v_perturb = self.ProjectFrom2DTo3D(slip_velocity, n_BA_W)

        # Get the contact point in the robot frame.
        contact_point_W = robot_ball_contacts[0].p_WCa
        X_WA = all_plant.EvalBodyPoseInWorld(all_context, body_A)
        contact_point_A = X_WA.inverse().multiply(contact_point_W)

        # Get the contact Jacobian for the robot.
        body_A_in_robot_plant = self.controller.robot_plant.GetBodyByName(body_A.name())
        J_WAc = self.controller.robot_plant.CalcPointsGeometricJacobianExpressedInWorld(
            self.controller.robot_context, body_A_in_robot_plant.body_frame(), contact_point_A)

        # Use the Jacobian pseudo-inverse to transform the velocity perturbation
        # to the change in robot generalized velocity.
        v_robot_perturb, residuals, rank, singular_values = np.linalg.lstsq(J_WAc, v_perturb)

        # Prepare to simulate the system forward by one controller time cycle, using time and
        # configuration set from time t and the newly perturbed configuration.
        simulator = Simulator(self.diagram)
        simulator.set_publish_every_time_step(True)
        context = simulator.get_mutable_context()
        mbw_context = self.diagram.GetMutableSubsystemContext(self.mbw, context)
        robot_and_ball_context = self.mbw.GetMutableSubsystemContext(all_plant, mbw_context)
        plan = self.controller.plan
        context.set_time(t)
        robot_and_ball_state = robot_and_ball_context.get_mutable_state()
        robot_and_ball_state = all_context.get_state()
        all_plant.SetVelocities(robot_and_ball_context, self.controller.robot_instance, v_robot_perturb)

        # Step the simulation forward in time.
        dt = 1e-3
        simulator.StepTo(t + dt)

        # Get the new robot configuration.
        qnew = all_plant.GetPositions(robot_and_ball_context)
        vnew = all_plant.GetVelocities(robot_and_ball_context)
        newcontacts = self.controller.FindContacts(qnew)

        # Check whether the ball is still contacting the ground.
        if (not self.controller.IsBallContactingGround(contacts)):
            return [-1, 0]

        # Check whether the ball and robot are still contacting.
        if (not self.controller.IsRobotContactingBall(contacts)):
            return [-2, 0]

        # If still here, all contacts are still maintained. Compute the slip
        # velocities for the ball/ground and ball/robot.
        [N, S, T, Ndot, Sdot, Vdot] = self.controller.ConstructJacobians(newcontacts, qnew)

        # Compute the change in slip velocities.
        Sv = S.dot(v)
        Tv = T.dot(v)
        new_slip_norm = np.linalg.norm([np.linalg.norm(Sv), np.linalg.norm(Tv)])
        return [0, new_slip_norm / np.linalg.norm(slip_velocity)]


    # Evaluates the ability of the controller to deal with slip at all points in
    # time.
    def EvaluateSlipPerturbation(self):
        # Get the plan.
        plan = self.controller.plan
        t_final = plan.end_time()

        # Open file for writing.
        handle = open('slip_eval.dat', 'w')

        # Advance time, finding a point at which contact is desired.
        dt = 1e-3
        t = 0.0
        while t <= t_final:
            # Determine whether the plan indicates contact is desired.
            if not plan.IsContactDesired(t):
                t += dt
                continue

            # Get the contacts.
            q, v = self.SetStates(t)
            contacts = self.controller.FindContacts(q)

            # Verify that the robot is contacting the ball.
            if (not self.controller.IsRobotContactingBall(contacts)):
                print 'Expected the robot to be contacting the ball at time ' + str(t) + ' but it is not.'
                t += dt
                continue

            # Evaluate.
            output = self.EvaluateSlipPerturbationAtTime(t)
            handle.write(str(t) + ': ' + str(output[0]) + ' ' + str(output[1]) + '\n')
            handle.flush()

            # Update t.
            t += dt

        # Close the file- all done!
        handle.close();

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time_step", type=float, default=0.,
        help="If greater than zero, the plant is modeled as a system with "
             "discrete updates and period equal to this time_step. "
             "If 0, the plant is modeled as a continuous system.")
    parser.add_argument("--kp", type=float, default=60.0,
        help="Cartesian Kp for impedance control. Gets used for all xyz "
             "directions.")
    parser.add_argument("--kd", type=float, default=30.0,
        help="Cartesian kd for impedance control. Gets used for all xyz "
             "directions.")
    args = parser.parse_args()

    # Gains in Cartesian-land.
    robot_cart_kp = np.ones([3, 1]) * args.kp
    robot_cart_kd = np.ones([3, 1]) * args.kd

    # Joint gains for the robot.
    nv_robot = 6
    robot_gv_kp = np.ones([nv_robot, 1]) * 10
    robot_gv_ki = np.ones([nv_robot, 1]) * 0.1
    robot_gv_kd = np.ones([nv_robot, 1]) * 1.0

    # Construct and run the evaluator.
    bce = BoxControllerEvaluator(args.time_step, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd)
    bce.EvaluateSlipPerturbation()


if __name__ == "__main__":
    main()
