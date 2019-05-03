import math
import sys
import numpy as np
import unittest
import argparse
import logging
from recordclass import recordclass

from manipulation_plan import ManipulationPlan
from box_controller import BoxController
from box_soccer_simulation import BuildBlockDiagram
from embedded_box_soccer_sim import EmbeddedSim

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType, BasicVector, MultibodyForces,
                          ComputeBasisFromAxis, MathematicalProgram)
from pydrake.solvers import mathematicalprogram

class ControllerTest(unittest.TestCase):
    def setUp(self):
      self.controller, self.diagram, self.all_plant, self.robot_plant, self.mbw, self.robot_instance, self.ball_instance, robot_continuous_state_output_port = BuildBlockDiagram(self.step_size, self.penetration_allowance, self.plan_path, self.fully_actuated)

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
      plant.SetPositions(self.controller.robot_and_ball_context, q)
      plant.SetVelocities(self.controller.robot_and_ball_context, v)

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

    # This function outputs contact information.
    def PrintContacts(self, q):
      # Get contacts.
      contacts = self.controller.FindContacts(q)

      # Get the inspector.
      robot_and_ball_context = self.controller.robot_and_ball_context
      query_object = self.all_plant.EvalAbstractInput(
        robot_and_ball_context, self.controller.geometry_query_input_port.get_index()).get_value()
      inspector = query_object.inspector()

      for i in contacts:
        geometry_A_id = i.id_A
        geometry_B_id = i.id_B
        frame_A_id = inspector.GetFrameId(geometry_A_id)
        frame_B_id = inspector.GetFrameId(geometry_B_id)
        body_A = self.all_plant.GetBodyFromFrameId(frame_A_id)
        body_B = self.all_plant.GetBodyFromFrameId(frame_B_id)
        logging.info('Contact found between ' + body_A.name() + ' and ' + body_B.name() + ' at: ' + str(i.p_WCa))
        logging.info('  Point of contact: ' + str(0.5 * (i.p_WCa + i.p_WCb)))
        logging.info('  Normal (pointing to ' + body_A.name() + '): ' + str(i.nhat_BA_W))

    # Tests that ComputeContributionsFromPlannedBoxPosition() produces control forces that accelerate in the direction
    # of the planned error in position.
    def test_ComputeContributionsFromPlannedBoxPosition(self):
        # Generate a configuration for the box corresponding to a rotation about z (so that the narrowest dimension
        # points along the x-axis) and vertical translation to have the box "rest" on the ground.
        # NOTE: we will leave the box configuration to NaN to "prove" that the box location isn't accounted for in any
        # calculations.
        box_height = 0.4
        sqrt2_2 = math.sqrt(2)/2.0
        q_box = np.array([sqrt2_2, 0, 0, sqrt2_2, 0, 0, box_height/2])
        q0 = np.ones([self.all_plant.num_positions()]) * float('nan')
        self.all_plant.SetPositionsInArray(self.controller.robot_instance, q_box, q0)

        # Set the planned position of the box along the x-axis.
        q_box_planned = np.array([sqrt2_2, 0, 0, sqrt2_2, 1.0, 0, box_height/2])

        # Curate the signed distance data, first to point exactly along the error from the plan.
        normal_and_signed_distance = recordclass(
                'NormalAndSignedDistanceData', 'phi normal_foot_ball_W closest_foot_body')
        normal_and_signed_distance.phi = float('nan')
        normal_and_signed_distance.normal_foot_ball_W = np.array([1, 0, 0])
        normal_and_signed_distance.closest_foot_body = self.controller.robot_and_ball_plant.GetBodyByName("box")

        # Compute the change in q (i.e., desired v) from the error in the planned position of the box. This should be
        # zero because the planned error is aligned with the normal direction.
        v_from_planned_position = self.controller.ComputeContributionsFromPlannedBoxPosition(
            q0, normal_and_signed_distance, q_box_planned)
        self.assertAlmostEqual(np.linalg.norm(v_from_planned_position), 0.0)

        # Now change the normal in the signed distance data to point completely orthogonally from the plan.
        normal_and_signed_distance.normal_foot_ball_W = np.array([0, 1, 0]) 

        # Compute the change in q (i.e., desired v) from the error in the planned position of the box. The component of
        # linear translation in the x-direction should be nonzero. Every other component should be zero.
        v_from_planned_position = self.controller.ComputeContributionsFromPlannedBoxPosition(
            q0, normal_and_signed_distance, q_box_planned)
        self.assertGreater(v_from_planned_position[3], 5e-1)
        self.assertAlmostEqual(np.linalg.norm(v_from_planned_position[0:3]), 0)  # Angular velocity.
        self.assertAlmostEqual(np.linalg.norm(v_from_planned_position[4:6]), 0)  # y/z translational velocity.

    # Tests that ComputeContributionsFromPlannedBoxVelocity() produces control forces that accelerate in the direction
    # of the planned error in position.
    def test_ComputeContributionsFromPlannedBoxVelocity(self):
        # Generate a configuration for the box corresponding to a rotation about z (so that the narrowest dimension
        # points along the x-axis) and vertical translation to have the box "rest" on the ground.
        # NOTE: we will leave the box configuration to NaN to "prove" that the box location isn't accounted for in any
        # calculations.
        box_height = 0.4
        sqrt2_2 = math.sqrt(2)/2.0
        q_box = np.array([sqrt2_2, 0, 0, sqrt2_2, 0, 0, box_height/2])
        q0 = np.ones([self.all_plant.num_positions()]) * float('nan')
        self.all_plant.SetPositionsInArray(self.controller.robot_instance, q_box, q0)

        # Set the planned position of the box to its current position.
        q_box_planned = q_box

        # Set the planned velocity of the box along the x-axis, and the current box velocity to zero.
        v0 = np.ones([self.all_plant.num_velocities()]) * float('nan')
        v_box_planned = np.array([0, 0, 0, 1.0, 0, 0])
        v_box = np.zeros([self.robot_plant.num_velocities()])
        self.all_plant.SetVelocitiesInArray(self.controller.robot_instance, v_box, v0)

        # Curate the signed distance data, first to point exactly along the error from the plan.
        normal_and_signed_distance = recordclass(
                'NormalAndSignedDistanceData', 'phi normal_foot_ball_W closest_foot_body')
        normal_and_signed_distance.phi = float('nan')
        normal_and_signed_distance.normal_foot_ball_W = np.array([1, 0, 0])
        normal_and_signed_distance.closest_foot_body = self.controller.robot_and_ball_plant.GetBodyByName("box")

        # Compute the desired v from the error in the planned velocity of the box. This should be
        # zero because the planned error is aligned with the normal direction.
        v_from_planned_velocity = self.controller.ComputeContributionsFromPlannedBoxVelocity(
            q0, v0, normal_and_signed_distance, q_box_planned, v_box_planned)
        self.assertAlmostEqual(np.linalg.norm(v_from_planned_velocity), 0.0)

        # Now change the normal in the signed distance data to point completely orthogonally from the plan.
        normal_and_signed_distance.normal_foot_ball_W = np.array([0, 1, 0]) 

        # Compute the desired v from the error in the planned velocity of the box. The component of
        # linear translation in the x-direction should be nonzero. Every other component should be zero.
        v_from_planned_velocity = self.controller.ComputeContributionsFromPlannedBoxVelocity(
            q0, v0, normal_and_signed_distance, q_box_planned, v_box_planned)
        self.assertGreater(v_from_planned_velocity[3], 5e-1)
        self.assertAlmostEqual(np.linalg.norm(v_from_planned_velocity[0:3]), 0)  # Angular velocity.
        self.assertAlmostEqual(np.linalg.norm(v_from_planned_velocity[4:6]), 0)  # y/z translational velocity.

    def test_ComputeContributionsFromBallTrackingErrors(self):
        # Generate a configuration for the box. The configuration will correspond to a rotation about z (so that the
        # narrowest dimension is facing the ball) and vertical translation to have the box "rest" on the ground.
        box_height = 0.4
        sqrt2_2 = math.sqrt(2)/2.0
        q_box = np.array([sqrt2_2, 0, 0, sqrt2_2, 0, 0, box_height/2])
        q0 = np.ones([self.all_plant.num_positions()]) * float('inf')
        self.all_plant.SetPositionsInArray(self.controller.robot_instance, q_box, q0)

        # Generate a configuration for the ball such that the box and the ball are separated along the x-axis by a
        # pre-specified distance (1.0).
        box_depth = 0.04
        ball_radius = 0.1
        q_ball = np.array([1, 0, 0, 0, 1.0 + box_depth / 2 + ball_radius, 0, ball_radius])
        self.all_plant.SetPositionsInArray(self.controller.ball_instance, q_ball, q0)

        # Set the velocities to zero initially.
        v0 = np.zeros([self.all_plant.num_velocities()])        

        # Curate the signed distance data, to point exactly toward the ball position.
        normal_and_signed_distance = recordclass(
                'NormalAndSignedDistanceData', 'phi normal_foot_ball_W closest_foot_body')
        normal_and_signed_distance.phi = 1.0
        normal_and_signed_distance.normal_foot_ball_W = np.array([1, 0, 0])
        normal_and_signed_distance.closest_foot_body = self.controller.robot_and_ball_plant.GetBodyByName("box")

        # Compute the desired v from the error in the ball position. The component of linear translation in the
        # x-direction should be nonzero. Every other component should be zero.
        [v_from_position_ball_tracking, v_from_velocity_ball_tracking] = \
                self.controller.ComputeContributionsFromBallTrackingErrors(q0, v0, normal_and_signed_distance)
        self.assertAlmostEqual(np.linalg.norm(v_from_velocity_ball_tracking), 0)
        self.assertGreater(v_from_position_ball_tracking[3], 5e-1)
        self.assertAlmostEqual(np.linalg.norm(v_from_position_ball_tracking[0:3]), 0)  # Angular velocity.
        self.assertAlmostEqual(np.linalg.norm(v_from_position_ball_tracking[4:6]), 0)  # y/z translational velocity.

        # Indicate that there is no error in signed distance, and that there is an error in relative velocity.
        normal_and_signed_distance.phi = self.controller.get_desired_ball_signed_distance_under_contact()
        self.all_plant.SetVelocitiesInArray(self.controller.ball_instance, [0, 0, 0, 1, 0, 0], v0)

        # Compute the desired v from the error in the ball velocity. The component of linear translation in the
        # x-direction should be nonzero. Every other component should be zero.
        [v_from_position_ball_tracking, v_from_velocity_ball_tracking] = \
                self.controller.ComputeContributionsFromBallTrackingErrors(q0, v0, normal_and_signed_distance)
        self.assertAlmostEqual(np.linalg.norm(v_from_position_ball_tracking), 0)
        self.assertGreater(v_from_velocity_ball_tracking[3], 5e-1)
        self.assertAlmostEqual(np.linalg.norm(v_from_velocity_ball_tracking[0:3]), 0)  # Angular velocity.
        self.assertAlmostEqual(np.linalg.norm(v_from_velocity_ball_tracking[4:6]), 0)  # y/z translational velocity.

    # Tests that GetNormalAndSignedDistanceFromRobotToBall() returns the correct result when the box and the foot
    # are disjoint, kissing, and interpenetrated.
    def test_GetNormalAndSignedDistanceFromRobotToBall(self):
        # Generate a configuration for the box- its location will not change during the following tests.
        # The configuration will correspond to a rotation about z (so that the narrowest dimension is facing the ball)
        # and vertical translation to have the box "rest" on the ground.
        box_height = 0.4
        sqrt2_2 = math.sqrt(2)/2.0
        q_box = np.array([sqrt2_2, 0, 0, sqrt2_2, 0, 0, box_height/2])
        q = np.ones([self.all_plant.num_positions()]) * float('inf')
        self.all_plant.SetPositionsInArray(self.controller.robot_instance, q_box, q)

        def GenerateBallConfigurationFromSignedDistance(target_signed_distance):
            box_depth = 0.04
            ball_radius = 0.1
            return np.array([1, 0, 0, 0, target_signed_distance + box_depth / 2 + ball_radius, 0, ball_radius])

        # Generate a configuration for the ball such that the box and the ball are separated along the x-axis by a
        # pre-specified distance.
        target_signed_distance = 1.0
        q_ball = GenerateBallConfigurationFromSignedDistance(target_signed_distance)
        self.all_plant.SetPositionsInArray(self.controller.ball_instance, q_ball, q)

        # Check the distance, normal, and body.
        normal_and_signed_distance_data = self.controller.GetNormalAndSignedDistanceFromRobotToBall(q)
        self.assertAlmostEqual(np.inner(normal_and_signed_distance_data.normal_foot_ball_W, np.array([1, 0, 0])), 1.0)
        self.assertAlmostEqual(normal_and_signed_distance_data.phi, target_signed_distance)
        self.assertEqual(normal_and_signed_distance_data.closest_foot_body.name(), 'box')

        # Generate a configuration such that the box and the ball are overlapping.
        target_signed_distance = -1e-3
        q_ball = GenerateBallConfigurationFromSignedDistance(target_signed_distance)
        self.all_plant.SetPositionsInArray(self.controller.ball_instance, q_ball, q)

        # Check the distance, normal, and body.
        normal_and_signed_distance_data = self.controller.GetNormalAndSignedDistanceFromRobotToBall(q)
        self.assertAlmostEqual(np.inner(normal_and_signed_distance_data.normal_foot_ball_W, np.array([1, 0, 0])), 1.0)
        self.assertAlmostEqual(normal_and_signed_distance_data.phi, target_signed_distance)
        self.assertEqual(normal_and_signed_distance_data.closest_foot_body.name(), 'box')       

        # Generate a configuration such that the box and the ball are kissing.
        target_signed_distance = 0.0
        q_ball = GenerateBallConfigurationFromSignedDistance(target_signed_distance)
        self.all_plant.SetPositionsInArray(self.controller.ball_instance, q_ball, q)

        # Check the distance, normal, and body.
        normal_and_signed_distance_data = self.controller.GetNormalAndSignedDistanceFromRobotToBall(q)
        self.assertAlmostEqual(np.inner(normal_and_signed_distance_data.normal_foot_ball_W, np.array([1, 0, 0])), 1.0)
        self.assertAlmostEqual(normal_and_signed_distance_data.phi, target_signed_distance)
        self.assertEqual(normal_and_signed_distance_data.closest_foot_body.name(), 'box')       


    '''
    # Tests that the acceleration of the box is exactly that planned when there is no error in the plan and
    # no error in ball/box overlap or overlap velocity.
    def test_TrackBoxAccelWhenNoError(self):

        # Set the ball position and velocity such that exactly the right amount of interpenetration and
        # relative velocity (between the box and the ball) are present. 

        # Ensure that the controls give exactly the acceleration desired.

    # Tests that the acceleration of the box is toward the ball when the error in the plan and the error in the
    # box/ball overlap are identical and the planned acceleration is zero. 

    # Tests that the acceleration of the box is zero when the planned acceleration is zero, the box is
    # overlapping with the ball the desired amount, and the planned box position or velocity is aligned with the
    # normal direction. 
    '''

    # This tests the control matrix for the robot (box).
    def test_ControlMatrix(self):

      # Get the control matrix.
      B = self.controller.ConstructRobotActuationMatrix()

      # Set a recognizable velocity vector.
      nv = self.controller.nv_robot() + self.controller.nv_ball()
      v = np.zeros([nv])
      v[:] = np.linspace(1, 10, nv)

      # Copy the elements corresponding to the robot out of v.
      vselected = self.all_plant.GetVelocitiesFromArray(self.controller.robot_instance, v)
      v_all_but_zero = np.zeros([nv])
      self.all_plant.SetVelocitiesInArray(self.controller.robot_instance, vselected, v_all_but_zero)
      vselected_row = np.zeros([len(vselected), 1])
      vselected_row[:,0] = vselected

      # Check whether B*v = v_all_but_zero.
      self.assertTrue(np.allclose(v_all_but_zero, B.dot(vselected).flatten()))

    # NOTE: this unit test is disabled because it's yet unclear whether we want
    # some zero entries corresponding to the ball angular velocity in the
    # weighting matrix.
    @unittest.expectedFailure
    # This tests the velocity weighting matrix for the ball.
    def test_BallVelocityMatrix(self):
      # Get the state weighting matrix.
      P = self.controller.ConstructBallVelocityWeightingMatrix()

      # Set a recognizable velocity vector.
      nv = self.controller.nv_robot() + self.controller.nv_ball()
      v = np.zeros([nv])
      v[:] = np.linspace(1, 10, nv)

      # Copy the elements corresponding to the ball out of v.
      vselected = np.reshape(self.all_plant.GetVelocitiesFromArray(self.controller.ball_instance, v), [-1, 1])
      v_all_but_zero = np.zeros([nv])
      self.all_plant.SetVelocitiesInArray(self.controller.ball_instance, vselected, v_all_but_zero)
      vselected_row = np.zeros([len(vselected), 1])
      vselected_row[:] = vselected

      # Check whether P*v = v_all_but_zero.
      self.assertTrue(np.allclose(v_all_but_zero, P.dot(vselected).flatten()))

    # This tests only that q and v for the ball and the robot (box) have
    # expected values.
    def test_QAndVSetProperly(self):
      t = 0  # Set the desired plan time.
      self.SetStates(t)

      # Get q_ball, v_ball, q_robot, and v_robot, as set in the inputs.
      q_robot = self.controller.get_q_robot(self.controller_context)
      v_robot = self.controller.get_v_robot(self.controller_context)
      q_ball = self.controller.get_q_ball(self.controller_context)
      v_ball = self.controller.get_v_ball(self.controller_context)

      # Get commands.
      q_robot_des = self.controller.plan.GetRobotQVAndVdot(t)[0:self.controller.nq_robot()]
      v_robot_des = self.controller.plan.GetRobotQVAndVdot(t)[self.controller.nq_robot():self.controller.nq_robot()+self.controller.nv_robot()]
      q_ball_des = self.controller.plan.GetBallQVAndVdot(t)[0:self.controller.nq_ball()]
      v_ball_des = self.controller.plan.GetBallQVAndVdot(t)[self.controller.nq_ball():self.controller.nq_ball()+self.controller.nv_ball()]

      # Verify that they're all close.
      self.assertTrue(np.allclose(q_robot, q_robot_des.flatten()))
      self.assertTrue(np.allclose(v_robot, v_robot_des.flatten()))
      self.assertTrue(np.allclose(q_ball, q_ball_des.flatten()))
      self.assertTrue(np.allclose(v_ball, v_ball_des.flatten()))

      # Verify that q is set in the proper place in the array.
      q_all = self.controller.get_q_all(self.controller_context)
      self.assertTrue(np.allclose(q_robot, self.controller.robot_and_ball_plant.GetPositionsFromArray(self.controller.robot_instance, q_all)))
      self.assertTrue(np.allclose(q_ball, self.controller.robot_and_ball_plant.GetPositionsFromArray(self.controller.ball_instance, q_all)))

    def test_BodyAccessors(self):
      # This is just a smoke test.
      self.controller.get_ball_from_robot_and_ball_plant()
      self.controller.get_foot_links_from_robot_and_ball_plant()
      self.controller.get_ground_from_robot_and_ball_plant()

    def test_IntegralValue(self):
      int_vec = np.linspace(0, 1, self.controller.nq_robot())
      self.controller.set_integral_value(self.controller_context, int_vec)
      self.assertTrue(np.allclose(self.controller.get_integral_value(self.controller_context), int_vec))

    def test_StateSizes(self):
      self.assertEqual(self.controller.nq_ball(), 7)
      self.assertEqual(self.controller.nv_ball(), 6)
      self.assertEqual(self.controller.nq_robot(), 7)
      self.assertEqual(self.controller.nv_robot(), 6)

    # Computes the contact force solution for test_NoSlipAcceleration (i.e., while also minimizing the deviation from
    # the desired acceleration). Compare to ComputeContactForcesWithoutControl().
    def ComputeContactForces(self, q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v, P, vdot_ball_des, enforce_no_slip=False):
        # Get the robot/ball plant and the correpsonding context from the controller.
        all_plant = self.controller.robot_and_ball_plant
        all_context = self.controller.robot_and_ball_context

        # Get everything we need to compute contact forces.
        all_plant.SetPositions(all_context, q)
        all_plant.SetVelocities(all_context, v)
        M = all_plant.CalcMassMatrixViaInverseDynamics(all_context)
        iM = np.linalg.inv(M)
        link_wrenches = MultibodyForces(all_plant)
        all_plant.CalcForceElementsContribution(all_context, link_wrenches)
        fext = -all_plant.CalcInverseDynamics(all_context, np.zeros([len(v)]), link_wrenches)
        fext = np.reshape(fext, [len(v), 1])

        # Solve the following QP:
        # fz = argmin 1/2 (vdot_des - P*vdot)' * (vdot_des - P * vdot)
        # subject to: M * vdot = f + N' * fn + S' * fs + T' * ft
        # fn >= 0
        #
        # where *we consider only the ball, not the robot.*
        #
        # The objective function expands to (leaving out vdot_des'*vdot_des term):
        # 1/2 vdot' * P' * P * vdot - vdot_des' * P * vdot =
        #   1/2 (fext' + fz' * Z) * inv(M) * P' * P * inv(M) * (fext + Z * fz) - vdot_des' * P * inv(M) * (fext + Z' * fz)
        #
        # The gradient of the above is: 1/2 * Z * inv(M) * P' * P * inv(M) * Z * fz - Z * inv(M) * P' * vdot_des +
        #    Z * inv(M) * P' * P * inv(M) * fext
        #
        # The Hessian matrix is: 1/2 * Z * inv(M) * P' * P * inv(M) * Z

        # Form the 'Z' matrix.
        [Z, Zdot_v] = self.controller.SetZAndZdot_v(N, S, T, Ndot_v, Sdot_v, Tdot_v)

        # Compute the Hessian.
        H = Z.dot(iM).dot(P.T).dot(P).dot(iM).dot(Z.T)

        # Compute the linear term.
        c = Z.dot(iM).dot(P.T).dot(P.dot(iM).dot(fext) - vdot_ball_des)

        # Formulate the QP.
        prog = mathematicalprogram.MathematicalProgram()
        vars = prog.NewContinuousVariables(len(c), "vars")
        prog.AddQuadraticCost(H, c, vars)
        nc = Z.shape[0]/3
        # Add a compressive-force constraint on the normal contact forces.
        lb = np.zeros(nc * 3)
        lb[nc:] = np.ones(nc * 2) * -float('inf')
        ub = np.ones(nc * 3) * float('inf')
        prog.AddBoundingBoxConstraint(lb, ub, vars)

        # There should be no contact separation in the normal direction.
        # N*vdot + Ndot_v = 0 ==> N(inv(M) * (f + Z'*fz)) + Ndot_v = 0, etc.
        rhs_n = -N.dot(iM).dot(fext) - Ndot_v
        prog.AddLinearConstraint(N.dot(iM).dot(Z.T), rhs_n, rhs_n, vars)

        # See whether to enforce the no-acceleration-in-contact directions conditions:
        # S*vdot + Sdot_v = 0 ==> S(inv(M) * (f + Z'*fz)) + Sdot_v = 0, etc.
        # T*vdot + Sdot_v = 0 ==> S(inv(M) * (f + Z'*fz)) + Sdot_v = 0, etc.
        if enforce_no_slip == True:
          rhs_s = -S.dot(iM).dot(fext) - Sdot_v
          prog.AddLinearConstraint(S.dot(iM).dot(Z.T), rhs_s, rhs_s, vars)
          rhs_t = -T.dot(iM).dot(fext) - Tdot_v
          prog.AddLinearConstraint(T.dot(iM).dot(Z.T), rhs_t, rhs_t, vars)

        # Solve the QP.
        result = prog.Solve()
        self.assertEquals(result, mathematicalprogram.SolutionResult.kSolutionFound, msg='Unable to solve QP, reason: ' + str(result))
        fz = np.reshape(prog.GetSolution(vars), [-1, 1])
        return [fz, Z, iM, fext]

    # Modifies the planned configuration to a new configuration that satisfies the target distance to the requisite
    # tolerance.
    # Returns the new plant configuration.
    def ModifyPlanForNecessaryDistance(self, q, target_dist, tol):
        logging.info('target distance: ' + str(target_dist))
        assert target_dist < 0
        assert tol > 0

        # Returns the Jacobian matrix for the closest point, projected along the given vector.
        def CalcJacobian(q_current, witness_point_W, projection_vector_W, body):
            # Get the geometric Jacobian for the velocity of the witness point point as moving with body.
            logging.debug('projection vector: ' + str(projection_vector_W))
            self.controller.robot_and_ball_plant.SetPositions(self.controller.robot_and_ball_context, q_current)
            J_Ww = self.controller.robot_and_ball_plant.CalcPointsGeometricJacobianExpressedInWorld(
                  self.controller.robot_and_ball_context, body.body_frame(), witness_point_W)
            return projection_vector_W.T.dot(J_Ww)

        # Computes the closest points between the two bodies.
        # Returns (1) the distance; (2) the direction between the witness points (pointing from B to A), expressed in
        # the world frame; and (3) the midpoint between the witness points as an offset vector expressed in the world
        # frame.
        def FindClosestPoints(q_current, body_A, body_B):
            all_plant = self.controller.robot_and_ball_plant
            all_context = self.controller.robot_and_ball_context

            # Construct pairs of bodies.
            body_pair = self.controller.MakeSortedPair(body_A, body_B)

            # Update the context to use configuration q1 in the query. This will modify
            # the mbw context, used immediately below.
            self.controller.robot_and_ball_plant.SetPositions(self.controller.robot_and_ball_context, q_current)

            # Evaluate scene graph's output port, getting a SceneGraph reference.
            query_object = all_plant.EvalAbstractInput(all_context,
                    self.controller.geometry_query_input_port.get_index()).get_value()
            inspector = query_object.inspector()

            # Get the closest points and distinguish which is which.
            closest_points = query_object.ComputeSignedDistancePairwiseClosestPoints()

            # Process the closest points.
            for i in closest_points:
                geometry_A_id = i.id_A
                geometry_B_id = i.id_B
                frame_A_id = inspector.GetFrameId(geometry_A_id)
                frame_B_id = inspector.GetFrameId(geometry_B_id)
                cp_body_A = all_plant.GetBodyFromFrameId(frame_A_id)
                cp_body_B = all_plant.GetBodyFromFrameId(frame_B_id)
                if self.controller.MakeSortedPair(cp_body_A, cp_body_B) != body_pair:
                    continue
                X_WA = all_plant.EvalBodyPoseInWorld(all_context, cp_body_A)
                X_WB = all_plant.EvalBodyPoseInWorld(all_context, cp_body_B)
                closest_Aw = X_WA.multiply(i.p_ACa)
                closest_Bw = X_WB.multiply(i.p_BCb)
                if cp_body_A.name() == 'ground_body':
                    closest_Aw[2] = 0
                if cp_body_B.name() == 'ground_body':
                    closest_Bw[2] = 0
                dist = i.distance
                if dist >= 0:
                    witness_W = 0.5 * (closest_Aw + closest_Bw)
                    n_W = closest_Aw - closest_Bw
                    logging.debug('closest on A: ' + str(closest_Aw))
                    logging.debug('closest on B: ' + str(closest_Bw))
                    if np.linalg.norm(n_W) <= 1e-15:
                        n_W = np.array([0, 0, 1])
                    else:
                        n_W = n_W / np.linalg.norm(n_W)
                    if cp_body_A != body_A:
                        n_W = -n_W
                    return [ dist, n_W, witness_W ]

            # Distance *must* be less than zero.
            assert dist < 0

            # Get the contact between the bodies.
            contacts = self.controller.FindContacts(q_current)
            for i in contacts:
                geometry_A_id = i.id_A
                geometry_B_id = i.id_B
                frame_A_id = inspector.GetFrameId(geometry_A_id)
                frame_B_id = inspector.GetFrameId(geometry_B_id)
                contacting_body_A = all_plant.GetBodyFromFrameId(frame_A_id)
                contacting_body_B = all_plant.GetBodyFromFrameId(frame_B_id)
                if self.controller.MakeSortedPair(contacting_body_A, contacting_body_B) != body_pair:
                    continue
                witness_W = 0.5 * (i.p_WCa + i.p_WCb)
                n_W = i.nhat_BA_W
                if cp_body_A != body_A:
                    n_W = -n_W
                dist = -i.depth
                assert dist < 0
                return [ dist, n_W, witness_W ]


        # Get bodies.
        ball_body = self.controller.get_ball_from_robot_and_ball_plant()
        box_body_list = self.controller.get_foot_links_from_robot_and_ball_plant()
        assert len(box_body_list) == 1
        box_body = box_body_list[0]
        world_body = self.controller.get_ground_from_robot_and_ball_plant()

        # Set the Newton-Raphson scaling parameter.
        alpha = 0.1

        # Compute the initial ball/ground distance.
        ball_ground_dist, n_ball_ground_W, ball_ground_witness_W = FindClosestPoints(q, ball_body, world_body)
        logging.debug(' ball/ground distance: ' + str(ball_ground_dist))
        # This only works for the known good configuration.
        # self.assertGreater(np.inner(n_ball_ground_W, np.array([0, 0, 1])), 0.5)

        # Loop until the distance is satisfied within the tolerance.
        while ball_ground_dist > target_dist and abs(ball_ground_dist - target_dist) > tol:
            # Moving along the positive direction of the contact normal will cause the bodies to separate, and
            # moving along the negative direction will cause the bodies to overlap further.

            # Get the Jacobian matrix along the contact normal, N.
            N_ball = CalcJacobian(q, ball_ground_witness_W, n_ball_ground_W, ball_body)

            # delta_q_v = N' * (target_dist - dist) will move the ball pose toward the desired distance.
            delta_q_v_ball = N_ball * (target_dist - ball_ground_dist)

            # delta_q_v is the desired change in configuration, *expressed in velocity coordinates*. Convert the
            # velocity coordinates to change in configuration.
            delta_q = self.controller.robot_and_ball_plant.MapVelocityToQDot(
                    self.controller.robot_and_ball_context, delta_q_v_ball)

            # Update q.
            q = q + delta_q * alpha
            logging.debug('q: ' + str(q))
            logging.debug('ball q: ' + str(self.controller.robot_and_ball_plant.GetPositionsFromArray(
                    self.controller.ball_instance, q)))

            # Compute the new ball/ground distance.
            ball_ground_dist, n_ball_ground_W, ball_ground_witness_W = FindClosestPoints(q, ball_body, world_body)
            logging.debug(' ball/ground distance: ' + str(ball_ground_dist))
            # This only works for the known good configuration.
            # self.assertGreater(np.inner(n_ball_ground_W, np.array([0, 0, 1])), 0.5)

        # Now compute the ball/box distance.
        ball_box_dist, n_ball_box_W, ball_box_witness_W = FindClosestPoints(q, ball_body, box_body)
        # This only works for the known good configuration.
        #self.assertGreater(np.inner(n_ball_box_W, np.array([0, 0, -1])), 0.5)

        # Loop until the distance is satisfied within the tolerance.
        while ball_box_dist > target_dist and abs(ball_box_dist - target_dist) > tol:
            logging.debug('ball/box distance: ' + str(ball_box_dist))

            # Moving along the positive direction of the contact normal will cause the bodies to separate, and
            # moving along the negative direction will cause the bodies to overlap further.

            # Get the Jacobian matrx along the contact normals, N.
            N_box = CalcJacobian(q, ball_box_witness_W, n_ball_box_W, box_body)

            # delta_q_v = N' * (target_dist - dist) will move the box pose toward the desired distance.
            delta_q_v_box = -N_box * (target_dist - ball_box_dist)

            # delta_q_v is the desired change in configuration, *expressed in velocity coordinates*. Convert the
            # velocity coordinates to change in configuration.
            delta_q = self.controller.robot_and_ball_plant.MapVelocityToQDot(
                    self.controller.robot_and_ball_context, delta_q_v_box)

            # Update q.
            q = q + delta_q * alpha
            logging.debug('box q: ' + str(self.controller.robot_and_ball_plant.GetPositionsFromArray(
                    self.controller.robot_instance, q)))

            # Compute the new ball/box distance.
            ball_box_dist, n_ball_box_W, ball_box_witness_W = FindClosestPoints(q, ball_body, box_body)
            # This only works for the known good configuration.
            # self.assertGreater(np.inner(n_ball_box_W, np.array([0, 0, -1])), 0.5)

        return q

    # Tests the ability to modify a plan to satisfy the necessary distance constraint.
    def test_ModifyPlanForNecessaryDistance(self):
        # First compute the known good configuration with almost *no* overlap.
        q = self.ConstructKnownGoodConfiguration(overlap=1e-14)[0]

        # Set the target distance and the tolerance.
        target_dist = -self.penetration_allowance
        tol = self.penetration_allowance * 1e-3

        # Now modify the plan to satisfy a given distance.
        q = self.ModifyPlanForNecessaryDistance(q, target_dist, tol)

        # Check the distances between the bodies.
        self.controller.robot_and_ball_plant.SetPositions(self.controller.robot_and_ball_context, q)
        ball_ground_dist = self.controller.GetSignedDistanceFromBallToGround()
        ball_box_dist = self.controller.GetSignedDistanceFromRobotToBall()
        self.assertAlmostEqual(ball_ground_dist, target_dist, delta=tol)
        self.assertAlmostEqual(ball_box_dist, target_dist, delta=tol)

    # Constructs a known, good configuration for testing controllers, given the desired quantity of non-negative overlap
    # between the bodies. The configuration is such that the box is on top of the ball.
    # Returns (1) the planned q for all models, (2) the planned v for all models, (3) the desired ball acceleration, and
    # (4) the desired box acceleration.
    def ConstructKnownGoodConfiguration(self, overlap):
        all_plant = self.controller.robot_and_ball_plant
        all_context = self.controller.robot_and_ball_context
        nq = self.controller.nq_ball() + self.controller.nq_robot()
        q = np.zeros([nq])

        # Set q for the ball: it will lie "exactly" on the ground.
        r = 0.1   # Ball radius.
        q_ball = np.array([1, 0, 0, 0, 0, 0, r - overlap])
        all_plant.SetPositionsInArray(self.controller.ball_instance, q_ball, q)

        # Set q for the box: it will correspond to a rotation about x and some
        # translation.
        box_depth = 0.04
        sqrt2_2 = math.sqrt(2)/2.0
        q_box = np.array([sqrt2_2, -sqrt2_2, 0, 0, 0, 0, 2*r + box_depth/2 - 2*overlap])
        all_plant.SetPositionsInArray(self.controller.robot_instance, q_box, q)

        # v is zero.
        nv = self.controller.nv_ball() + self.controller.nv_robot()
        v = np.zeros([nv])

        # Construct the desired accelerations.
        vdot_ball_des = np.reshape(np.array([0, -1/r, 0, -1, 0, 0]), [-1, 1])
        vdot_box_des = np.reshape(np.array([0, 0, 0, -2, 0, 0]), [-1, 1])
        vdot_des = v.copy()
        all_plant.SetVelocitiesInArray(self.ball_instance, vdot_ball_des, vdot_des)
        all_plant.SetVelocitiesInArray(self.robot_instance, vdot_box_des, vdot_des)

        return [q, v, vdot_des]

    # Checks that the ball and the box can be accelerated as desired using a known, good configuration.
    def test_TrackAccelerationWithKnownGoodConfiguration(self):
        # Construct the weighting matrix.
        nv = self.controller.nv_ball() + self.controller.nv_robot()
        v = np.zeros([nv])
        dummy_ball_v = np.array([1, 1, 1, 1, 1, 1])
        self.controller.robot_and_ball_plant.SetVelocitiesInArray(self.controller.ball_instance, dummy_ball_v, v)
        P = np.zeros([6, nv])
        vball_index = 0
        for i in range(nv):
            if abs(v[i]) > 0.5:
                P[vball_index, i] = v[i]
                vball_index += 1

        # Check a known, good configuration. The penalty method time scale tells how much the bodies should overlap.
        all_plant = self.controller.robot_and_ball_plant
        q, v, vdot_des = self.ConstructKnownGoodConfiguration(overlap=1e-14)#self.penetration_allowance*500)
        self.PrintContacts(q)
        contacts = self.controller.FindContacts(q)
        N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)
        nc = N.shape[0]
        assert nc == 2
        [Z, Zdot_v] = self.controller.SetZAndZdot_v(N, S, T, Ndot_v, Sdot_v, Tdot_v)

        # Get the contact index for the ball and the ground.
        ball_ground_contact_index = self.controller.GetBallGroundContactIndex(q, contacts)
        S_ground = S[ball_ground_contact_index, :]
        T_ground = T[ball_ground_contact_index, :]
        Sdot_v_ground = Sdot_v[ball_ground_contact_index]
        Tdot_v_ground = Tdot_v[ball_ground_contact_index]

        # Get external forces and inertia matrix.
        all_context = self.controller.robot_and_ball_context
        all_plant.SetPositions(all_context, q)
        all_plant.SetVelocities(all_context, v)
        M = all_plant.CalcMassMatrixViaInverseDynamics(all_context)
        iM = np.linalg.inv(M)
        link_wrenches = MultibodyForces(all_plant)
        all_plant.CalcForceElementsContribution(all_context, link_wrenches)
        fext = -all_plant.CalcInverseDynamics(all_context, np.zeros([len(v)]), link_wrenches)
        fext = np.reshape(fext, [len(v), 1])

        # Verify that the no-slip controller arrives at the same objective (or better).
        vdot_ball_des = all_plant.GetVelocitiesFromArray(self.controller.ball_instance, vdot_des)
        P = self.controller.ConstructVelocityWeightingMatrix()
        B = self.controller.ConstructRobotActuationMatrix()
        '''
        u, fz = self.controller.ComputeContactControlMotorForcesNoSlip(iM, fext, vdot_ball_des[-3:], Z, Zdot_v, N, Ndot_v, S_ground, Sdot_v_ground, T_ground, Tdot_v_ground)
        logging.info('computed actuator forces: ' + str(u))
        logging.info('computed contact forces: ' + str(fz))
        vdot = iM.dot(fext + B.dot(u) + Z.T.dot(fz))

        # Get the force required to accelerate the box and the ball to these accelerations.
        nv = self.controller.nv_ball() + self.controller.nv_robot()
        vdot_des = np.zeros([nv, 1])
        all_plant.SetVelocitiesInArray(self.controller.ball_instance, vdot_ball_des, vdot_des)
        all_plant.SetVelocitiesInArray(self.controller.robot_instance, vdot_box_des, vdot_des)
        logging.info('Necessary forces: ' + str(M.dot(vdot_des) - fext))

        # Verify that the desired linear acceleration was achieved.
        self.assertAlmostEqual(np.linalg.norm(P_star.dot(vdot) - vdot_ball_des[-3:]), 0, places=4)

        # Verify that the desired spatial acceleration was achieved.
        #self.assertAlmostEqual(np.linalg.norm(P.dot(vdot) - vdot_ball_des), 0, places=3)
        '''

        # Determine the control forces using the black box controller.
        u = self.controller.ComputeOptimalContactControlMotorForces(self.controller_context, q, v, vdot_des, weighting_type='full')

        # Check the desired spatial acceleration in the embedded simulation.
        vdot_approx = self.controller.ComputeApproximateAcceleration(self.controller_context, q, v, u)

        # Get the actual and desired ball and box accelerations.
        vdot_box_des = all_plant.GetVelocitiesFromArray(self.controller.robot_instance, vdot_des)
        vdot_ball_approx = all_plant.GetVelocitiesFromArray(self.controller.ball_instance, vdot_approx)
        vdot_box_approx = all_plant.GetVelocitiesFromArray(self.controller.robot_instance, vdot_approx)
        logging.info('approximate ball vdot: ' + str(vdot_ball_approx))
        logging.info('desired ball vdot: ' + str(vdot_ball_des))
        logging.info('approximate box vdot: ' + str(vdot_box_approx))
        logging.info('desired box vdot: ' + str(vdot_box_des))
        logging.info('P (linear) * (vdot_des - vdot_approx) norm: ' +
                str(np.linalg.norm(self.controller.ConstructVelocityWeightingMatrix('ball-linear').dot(vdot_des -
                        vdot_approx))))

        self.assertAlmostEqual(np.linalg.norm(vdot_ball_des - vdot_ball_approx), 0, places=5)
        self.assertAlmostEqual(np.linalg.norm(vdot_box_des - vdot_box_approx), 0, places=5)

    # Checks that the ball can be accelerated according to the plan.
    def test_PlannedAccelerationTracking(self):
      # Construct the weighting matrix.
      weighting_type = 'full'
      P = self.controller.ConstructVelocityWeightingMatrix(weighting_type)

      # Get the plan.
      plan = self.controller.plan

      # Get the robot/ball plant and the correpsonding context from the controller.
      all_plant = self.controller.robot_and_ball_plant
      all_context = self.controller.robot_and_ball_context

      # Set tolerances.
      zero_velocity_tol = 1e-4
      zero_accel_tol = 1e-6
      complementarity_tol = 1e-6

      # Advance time, finding a point at which contact is desired *and* where
      # the robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = dt#plan.end_time()
      while t < t_final:
        # Keep looping if contact is not desired.
        if not plan.IsContactDesired:
          t += dt
          continue

        # Set the amount of overlap.
        # Note: This amount of overlap is not optimal, but seems to work well, particularly for small step sizes.
        overlap = 1e-14#2.5e-2
        # Note: This *should* be the right amount of overlap, but it doesn't give the rigid contact result.
        #overlap = self.penetration_allowance

        # Set the signed distance to what we need.
        q, v = self.SetStates(t)
        q = self.ModifyPlanForNecessaryDistance(q, -overlap, overlap * 1e-2)
        contacts = self.controller.FindContacts(q)

        logging.info('-- TestPlannedAccelerationTracking() - identified testable time/state at t=' + str(t))
        self.PrintContacts(q)

        # Get the desired acceleration.
        vdot_ball_des = plan.GetBallQVAndVdot(self.controller_context.get_time())[-self.controller.nv_ball():]
        vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])
        logging.debug('desired ball acceleration (angular/linear): ' + str(vdot_ball_des))

        # Get the Jacobians.
        N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)
        nc = N.shape[0]

        # Verify that the velocities in each direction of the contact frame are zero.
        # NOTE: this is disabled because it no longer works after perturbing the amount of overlap.
        Nv = N.dot(v)
        Sv = S.dot(v)
        Tv = T.dot(v)
        #self.assertLess(np.linalg.norm(Nv), zero_velocity_tol, msg='Failed at t='+str(t))
        #self.assertLess(np.linalg.norm(Sv), zero_velocity_tol, msg='Failed at t='+str(t))
        #self.assertLess(np.linalg.norm(Tv), zero_velocity_tol, msg='Failed at t='+str(t))

        # Determine the control forces using the black box controller.
        vdot_box_des = self.controller.plan.GetRobotQVAndVdot(t)[-6:]
        vdot_des = np.zeros([len(v), 1])
        all_plant.SetVelocitiesInArray(self.controller.robot_instance, vdot_box_des, vdot_des)
        all_plant.SetVelocitiesInArray(self.controller.ball_instance, vdot_ball_des, vdot_des)
        u = self.controller.ComputeOptimalContactControlMotorForces(self.controller_context, q, v, vdot_des, weighting_type)

        # Check the desired spatial acceleration in the embedded simulation.
        vdot_approx = self.controller.ComputeApproximateAcceleration(self.controller_context, q, v, u)
        logging.info('approximate vdot: ' + str(vdot_approx))
        logging.info('P (full) * (vdot_des - vdot_approx) norm: ' +
                str(np.linalg.norm(self.controller.ConstructVelocityWeightingMatrix('full').dot(vdot_des - vdot_approx))))
        self.assertAlmostEqual(np.linalg.norm(P.dot(vdot_approx - vdot_des)), 0, places=5)#, msg='Failed at t='+str(t))

        # Check the desired spatial acceleration for the box.
        vdot_box_approx = np.reshape(all_plant.GetVelocitiesFromArray(self.controller.robot_instance, vdot_approx), [-1, 1])
        logging.info('desired box spatial acceleration: ' + str(vdot_box_des))
        logging.info('approximate box spatial acceleration: ' + str(vdot_box_approx))


        #self.assertAlmostEqual(np.linalg.norm(vdot_box_approx - vdot_box_des), 0, delta=2e-1, msg='Failed at t='+str(t))

        t += dt

    @unittest.expectedFailure
    # Check control outputs for when contact is intended and robot and ball are
    # indeed in contact and verifies that the desired acceleration is attained.
    # This test is disabled b/c MBP's contact dynamics do not match the rigid
    # contact model.
    def test_AccelerationFromLearnedDynamicsControlCorrect(self):
      # Get the actuation matrix.
      B = self.controller.ConstructRobotActuationMatrix()

      # Get the plan.
      plan = self.controller.plan

      # Get the robot/ball plant and the correpsonding context from the controller.
      all_plant = self.controller.robot_and_ball_plant
      all_context = self.controller.robot_and_ball_context

      # Advance time, finding a point at which contact is desired *and* where
      # the robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = plan.end_time()
      while True:
        if plan.IsContactDesired(t):
          # Look for contact.
          q, v = self.SetStates(t)
          contacts = self.controller.FindContacts(q)
          if self.controller.IsRobotContactingBall(q, contacts):
            logging.info('-- TestAccelerationFromLearnedDynamicsControlCorrect() - desired time identified: ' + str(t))
            break

        # No contact desired or contact was found.
        t += dt
        assert t <= t_final

      # Prepare to compute the control forces.
      M = all_plant.CalcMassMatrixViaInverseDynamics(all_context)
      link_wrenches = MultibodyForces(all_plant)
      all_plant.CalcForceElementsContribution(all_context, link_wrenches)
      fext = -all_plant.CalcInverseDynamics(all_context, np.zeros([len(v)]), link_wrenches)
      fext = np.reshape(fext, [len(v), 1])
      vdot_ball_des = plan.GetBallQVAndVdot(self.controller_context.get_time())[-self.controller.nv_ball():]
      vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])[-3:]

      # Determine the control forces using the black box controller.
      u = self.ComputeOptimalContactControlMotorForces(self.controller_context, q, v, vdot_ball_des)

      # Get the approximate velocity.
      vdot_approx = self.controller.ComputeApproximateAcceleration(self.controller_context, q, v, u)
      vdot_ball_approx = all_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)[-3:]

      # Check the accelerations.
      self.assertAlmostEqual(np.linalg.norm(vdot_ball_approx - vdot_ball_des), 0, places=5)

    # Check control outputs for when contact is intended and robot and ball are
    # indeed in contact and verifies that acceleration is as desired. This method
    # uses full actuation so that we know that the desired accelerations are
    # capable of being met. Hence, the desired and actual accelerations must be
    # equal at the optimal solution. So this test validates that the objective
    # function used in the control function's QP is sound.
    def test_FullyActuatedAccelerationCorrect(self):
      # Make sure the plant has been setup as fully actuated.
      if not self.fully_actuated:
        print('test_FullyActuatedAccelerationCorrect() only works with --fully_actuated=True option.')
        return

      # Get the plan.
      plan = self.controller.plan

      # Get the robot/ball plant and the correpsonding context from the controller.
      all_plant = self.controller.robot_and_ball_plant
      robot_and_ball_context = self.controller.robot_and_ball_context

      # Advance time, finding a point at which contact is desired *and* where
      # the robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = plan.end_time()
      while True:
        if plan.IsContactDesired(t):
          # Look for contact.
          q, v = self.SetStates(t)
          contacts = self.controller.FindContacts(q)
          if self.controller.IsRobotContactingBall(q, contacts):
            logging.info('-- test_FullyActuatedAccelerationCorrect() - desired time identified: ' + str(t))
            break

        # No contact desired or contact was found.
        t += dt
        assert t <= t_final

      # This test will compute the control forces on the robot and the contact
      # forces on the ball. The computed contact forces on the ball will be used
      # to integrate the ball velocity forward in time. The control and computed
      # contact forces on the robot will be used to integrate the robot velocity
      # forward in time as well.

      # Determine the predicted forces due to contact.
      f_act = self.controller.ComputeFullyActuatedBallControlForces(self.controller_context)

      # Step the plant forward by a small time.
      sim = EmbeddedSim(self.step_size)
      sim.UpdateTime(t)
      sim.UpdatePlantPositions(q)
      sim.UpdatePlantVelocities(v)
      sim.ApplyControls(f_act)
      sim.Step()

      # Compute the approximate acceleration.
      vnew = sim.GetPlantVelocities()
      vdot_approx = np.reshape((vnew - v) / sim.delta_t, (-1, 1))
      vdot_ball_approx = all_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)

      # Get the desired acceleration for the ball.
      nv_ball = self.controller.nv_ball()
      vdot_ball_des = plan.GetBallQVAndVdot(t)[-self.controller.nv_ball():]

      # Check the accelerations.
      self.assertAlmostEqual(np.linalg.norm(vdot_ball_approx - vdot_ball_des), 0, places=5)

    # Check that contact Jacobian construction is correct.
    def test_JacobianConstruction(self):
      # Get the plan.
      plan = self.controller.plan

      # Advance time, finding a point at which contact is desired *and* where
      # the robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = plan.end_time()
      while True:
        if plan.IsContactDesired(t):
          # Look for contact.
          q, v = self.SetStates(t)
          contacts = self.controller.FindContacts(q)
          if self.controller.IsRobotContactingBall(q, contacts):
            break

        # No contact desired or contact between robot and ball not found.
        t += dt
        assert t < t_final

      # Setup debugging output.
      logging.debug('Robot velocity: ' + str(self.all_plant.GetVelocitiesFromArray(self.controller.robot_instance, v)))
      logging.debug('Ball velocity: ' + str(self.all_plant.GetVelocitiesFromArray(self.controller.ball_instance, v)))

      # Construct the Jacobian matrices using the controller function.
      [N, S, T, Ndot_v, Sdot_v, Tdot_v] = self.controller.ConstructJacobians(contacts, q, v)

      self.PrintContacts(q)
      # Set a time step.
      dt = 1e-5

      # Get the inspector.
      robot_and_ball_context = self.controller.robot_and_ball_context
      # Necessary for up-to-date distance/contact results.
      self.all_plant.SetPositions(robot_and_ball_context, q)
      query_object = self.all_plant.EvalAbstractInput(
          robot_and_ball_context, self.controller.geometry_query_input_port.get_index()).get_value()
      inspector = query_object.inspector()

      # Examine each point of contact.
      for i in range(len(contacts)):
        # Get the two bodies.
        point_pair = contacts[i]
        geometry_A_id = point_pair.id_A
        geometry_B_id = point_pair.id_B
        frame_A_id = inspector.GetFrameId(geometry_A_id)
        frame_B_id = inspector.GetFrameId(geometry_B_id)
        body_A = self.all_plant.GetBodyFromFrameId(frame_A_id)
        body_B = self.all_plant.GetBodyFromFrameId(frame_B_id)
        logging.info('Processing contact between ' + body_A.name() + '' and '' + body_B.name())

        # The Jacobians yield the instantaneous movement of a contact point along
        # the various directions. Determine the location of the contact point
        # dt into the future in the world frame.
        kXAxisIndex = 0
        # Note: implicit assumption below is that ComputeBasisFromAxis() was
        # also used to determine directions for S and T.
        R_WC = ComputeBasisFromAxis(kXAxisIndex, point_pair.nhat_BA_W)
        pdot_C = np.zeros([3, 1])
        pdot_C[0] = N[i,:].dot(v)
        pdot_C[1] = S[i,:].dot(v)
        pdot_C[2] = T[i,:].dot(v)
        pdot_W = R_WC.dot(pdot_C)

        # Get the components of the contact point in the body frames.
        self.all_plant.SetPositions(robot_and_ball_context, q)
        X_WA = self.all_plant.EvalBodyPoseInWorld(robot_and_ball_context, body_A)
        X_WB = self.all_plant.EvalBodyPoseInWorld(robot_and_ball_context, body_B)
        p_A = X_WA.inverse().multiply(point_pair.p_WCa)
        p_B = X_WB.inverse().multiply(point_pair.p_WCb)

        # Update the generalized coordinates of the multibodies using a first-order
        # approximation and the arbitrary velocity.
        qdot = self.all_plant.MapVelocityToQDot(robot_and_ball_context, v)
        qnew = q + dt*qdot
        logging.debug('World contact point on A: ' + str(point_pair.p_WCa) + '  on B: ' + str(point_pair.p_WCb))
        logging.debug('Body contact point on A: ' + str(p_A) + '  on B: ' + str(p_B))
        logging.debug('q_robot (old): ' + str(self.all_plant.GetPositionsFromArray(self.controller.robot_instance, q)))
        logging.debug('nq_robot (new): ' + str(self.all_plant.GetPositionsFromArray(self.controller.robot_instance, qnew)))
        logging.debug('qdot_robot: ' + str(self.all_plant.GetPositionsFromArray(self.controller.robot_instance, qdot)))
        logging.debug('q_ball (old): ' + str(self.all_plant.GetPositionsFromArray(self.controller.ball_instance, q)))
        logging.debug('q_ball (new): ' + str(self.all_plant.GetPositionsFromArray(self.controller.ball_instance, qnew)))
        logging.debug('qdot_ball: ' + str(self.all_plant.GetPositionsFromArray(self.controller.ball_instance, qdot)))
        self.all_plant.SetPositions(robot_and_ball_context, qnew)

        # Determine the new locations of the points on the bodies. The difference
        # in the points yields a finite difference approximation to the relative
        # velocity.
        X_WA_new = self.all_plant.EvalBodyPoseInWorld(robot_and_ball_context, body_A)
        X_WB_new = self.all_plant.EvalBodyPoseInWorld(robot_and_ball_context, body_B)

        # The *velocity* at a point of contact, C, measured in the global frame is
        # the limit as h -> 0 of the difference in point location between t and
        # t + h, divided by h.
        logging.debug('New point location p_W_A: ' + str(X_WA_new.multiply(p_A)))
        logging.debug('New point location p_W_B: ' + str(X_WB_new.multiply(p_B)))
        pdot_W_A_approx = (X_WA_new.multiply(p_A) - X_WA.multiply(p_A)) / dt
        pdot_W_B_approx = (X_WB_new.multiply(p_B) - X_WB.multiply(p_B)) / dt
        logging.debug('pdot_W_A (approx): ' + str(pdot_W_A_approx))
        logging.debug('pdot_W_B (approx): ' + str(pdot_W_B_approx))
        pdot_W_approx = pdot_W_A_approx - pdot_W_B_approx
        logging.debug('pdot_W (approx): ' + str(pdot_W_approx))

        # The Jacobian-determined contact point and the new contact point should
        # differ little.
        logging.debug('pdot_W (true): ' + str(pdot_W))
        self.assertLess(np.linalg.norm(pdot_W.flatten() - pdot_W_approx.flatten()), dt, msg='pdot - ~approx too large (>' + str(dt) + ')')

    # Checks that planned contact points are equivalent to contact points
    # returned by the collision detector.
    def test_PlannedRobotBallContactsCoincidentWithCollisionDetectionPoints(self):
      # Get the plan.
      plan = self.controller.plan

      # Advance time, finding a point at which contact is desired *and* where the
      # robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = plan.end_time()
      while t <= t_final:
        # Set the planned q and v.
        if plan.IsContactDesired(t):
          q, v = self.SetStates(t)

        # Look for robot/ball.
        contacts = self.controller.FindRobotBallContacts(q)
        if len(contacts) > 0:
          # Get the *planned* point of contact in the world.
          # Set the contact point to an arbitrary point in the world.
          p = plan.GetContactKinematics(t)[0:3]

          # Get the computed contact point location in the world.
          pr_WA = contacts[0].p_WCa
          pr_WB = contacts[0].p_WCb
          p_true = (pr_WA + pr_WB) * 0.5

          # Verify that the planned and actual contact points are collocated.
          dist_tol = 1e-8
          self.assertEqual(len(contacts), 1)  # How do we choose from > 1?
          dist = np.linalg.norm(p.flatten() - p_true.flatten())
          err_msg = 'Points are not collocated at time ' + str(t) + ', p planned: ' + str(p) + ', p from geometry engine: ' + str(p_true), 'distance: ' + str(dist)
          self.assertLess(dist, dist_tol, msg=err_msg)

        t += dt

    # Check that velocity at the ball/ground contact point remains sufficiently close to zero.
    def test_ZeroVelocityBallGroundContact(self):
      # Get the plan.
      plan = self.controller.plan

      # Advance time, finding a point at which contact is desired *and* where the
      # robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = plan.end_time()
      while t <= t_final:
        # Set the planned q and v.
        if plan.IsContactDesired(t):
          q, v = self.SetStates(t)

        # Look for robot/ball.
        contacts = self.controller.FindBallGroundContacts(q)
        if len(contacts) > 0:
          # Get the computed contact point location in the world.
          pr_WA = contacts[0].p_WCa
          pr_WB = contacts[0].p_WCb
          p_true = (pr_WA + pr_WB) * 0.5

          # Construct the Jacobian matrices using the controller function.
          N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)

          # Output all contacting bodies
          self.PrintContacts(q)

          # Verify that the velocity at the contact point is approximately zero.
          zero_velocity_tol = 1e-12
          Nv = N.dot(v)
          logging.debug('Nv: ' + str(Nv))
          self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
          break

        # No contact desired or exactly two contacts not found.
        t += dt
        assert t < t_final

    # Check that velocity at the contact point remains sufficiently close to zero.
    def test_ZeroVelocityAtRobotBallContact(self):
      # Get the plan.
      plan = self.controller.plan

      # Advance time, finding a point at which contact is desired *and* where the
      # robot is contacting the ball.
      dt = 1e-3
      t = 0.0
      t_final = plan.end_time()
      while t < t_final:
        # Set the planned q and v.
        if plan.IsContactDesired(t):
          q, v = self.SetStates(t)

        # Look for robot/ball.
        contacts = self.controller.FindRobotBallContacts(q)
        if len(contacts) > 0:
          # Construct the Jacobian matrices using the controller function.
          N, S, T, Ndot, Sdot, Tdot = self.controller.ConstructJacobians(contacts, q, v)

          # Output all contacting bodies
          self.PrintContacts(q)

          # Verify that the velocity at the contact points are approximately zero.
          zero_velocity_tol = 1e-10
          Nv = N.dot(v)
          Sv = S.dot(v)
          Tv = T.dot(v)
          logging.debug('Nv: ' + str(Nv))
          logging.debug('Sv: ' + str(Sv))
          logging.debug('Tv: ' + str(Tv))
          self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
          self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
          self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

        t += dt

    # Check that the *absolute* contact distance when the plan indicates contact
    # is desired always lies below a threshold.
    def test_ContactDistanceBelowThreshold(self):
      # Get the plan.
      plan = self.controller.plan
      t_final = plan.end_time()

      # Advance time, finding a point at which contact is desired.
      dt = 1e-3
      t = 0.0
      within_tolerance = True
      fail_message = ''
      while t <= t_final:
        # Determine whether the plan indicates contact is desired.
        if not plan.IsContactDesired(t):
          t += dt
          continue

        # Set the states to q and v.
        self.SetStates(t)

        # Check the distance between the robot foot and the ball.
        dist_thresh = 1e-6
        dist_robot_ball = self.controller.GetSignedDistanceFromRobotToBall()
        dist_ground_ball = self.controller.GetSignedDistanceFromBallToGround()
        if abs(dist_robot_ball) > dist_thresh:
          within_tolerance = False
          fail_message += 'Robot/ball contact desired at t='+str(t)+' but signed distance is large (' + str(dist_robot_ball) + ').'
        if abs(dist_ground_ball) > dist_thresh:
          within_tolerance = False
          fail_message += 'Ball/ground contact desired at t='+str(t)+' but signed distance is large (' + str(dist_ground_ball) + ').'

        # TODO: Delete this assertion (use the one outside of the loop) when building in release mode.
        self.assertTrue(within_tolerance, msg=fail_message)

        # Update t.
        t += dt

      self.assertTrue(within_tolerance, msg=fail_message)

# Attempts to parse a string as a Boolean value.
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--fully_actuated", type=str2bool, default=False)
  parser.add_argument(
      "--penetration_allowance", type=float, default=1e-8,
      help="The amount of interpenetration to allow in the simulation.")
  parser.add_argument(
      "--step_size", type=float, default=1e-3,
      help="If greater than zero, the plant is modeled as a system with "
           "discrete updates and period equal to this step size. "
           "If 0, the plant is modeled as a continuous system.")
  parser.add_argument(
      "--plan_path", default='plan_box_curve/',
      help='Path to the plan')
  parser.add_argument(
      "--log", default='none',
      help='Logging type: "none", "info", "warning", "debug"')
  parser.add_argument('remainder', nargs=argparse.REMAINDER)
  args = parser.parse_args()

  # Set the step size, plan path, and whether the ball is fully actuated.
  ControllerTest.fully_actuated = args.fully_actuated
  ControllerTest.step_size = args.step_size
  ControllerTest.plan_path = args.plan_path
  ControllerTest.penetration_allowance = args.penetration_allowance

  # Set the logging level.
  if args.log.upper() != 'NONE':
      numeric_level = getattr(logging, args.log.upper(), None)
      if not isinstance(numeric_level, int):
          raise ValueError('Invalid log level: %s' % args.log)
      logging.basicConfig(level=numeric_level)
  else:
      logging.disable(logging.CRITICAL)

  # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
  sys.argv[1:] = args.remainder

  unittest.main()

