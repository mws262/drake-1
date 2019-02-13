import math
import sys
import numpy as np
import unittest
import argparse
import logging
from manipulation_plan import ManipulationPlan
from box_controller import BoxController
from box_soccer_simulation import BuildBlockDiagram
from embedded_box_soccer_sim import EmbeddedSim

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType,
                         BasicVector, MultibodyForces, ComputeBasisFromAxis)

class ControllerTest(unittest.TestCase):
  def setUp(self):
    self.controller, self.diagram, self.all_plant, self.robot_plant, self.mbw, self.robot_instance, self.ball_instance, robot_continuous_state_output_port = BuildBlockDiagram(self.step_size, self.plan_path, self.fully_actuated)

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

  # This function outputs contact information.
  def PrintContacts(self, t):
    # Get contacts.
    q, v = self.SetStates(t)
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
      logging.info("Contact found between " + body_A.name() + " and " + body_B.name() + " at: " + str(i.p_WCa))

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

  # Check control outputs for when contact is intended and robot and ball are
  # indeed in contact and verifies that slip is not caused.
  def test_ContactAndContactIntendedOutputsDoNotCauseSlip(self):
    # Get the plan.
    plan = self.controller.plan

    # Get the robot/ball plant and the correpsonding context from the controller.
    all_plant = self.controller.robot_and_ball_plant
    robot_and_ball_context = self.controller.robot_and_ball_context

    dbg_out = '\n\nDebugging output below:'

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
        if self.controller.IsRobotContactingBall(contacts):
          dbg_out += '\n  -- TestContactAndContactIntendedOutputsCorrect() - desired time identified: ' + str(t)
          break

      # No contact desired or contact was found.
      t += dt
      assert t <= t_final

    # This test will compute the control forces on the robot and the contact
    # forces on the ball. The computed contact forces on the ball will be used
    # to integrate the ball velocity forward in time. The control and computed
    # contact forces on the robot will be used to integrate the robot velocity
    # forward in time as well. We'll then examine the contact point and ensure
    # that its velocity remains sufficiently near zero.

    # Clear the velocity so that the contact velocity at the contact point need
    # not compensate for nonzero initial velocity.
    v[:] = np.zeros([len(v)])
    all_plant.SetPositions(robot_and_ball_context, q)
    all_plant.SetVelocities(robot_and_ball_context, v)
    N, S, T, Ndot, Sdot, Tdot = self.controller.ConstructJacobians(contacts, q)
    zero_velocity_tol = 1e-3
    Nv = N.dot(v)
    Sv = S.dot(v)
    Tv = T.dot(v)
    self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
    self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
    self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

    # Determine the predicted forces due to contact.
    f_act, f_contact = self.controller.ComputeActuationForContactDesiredAndContacting(self.controller_context, contacts)

    # Use the controller output to determine the generalized acceleration of the
    # robot and the ball.
    M = all_plant.CalcMassMatrixViaInverseDynamics(robot_and_ball_context)
    link_wrenches = MultibodyForces(self.all_plant)
    fext = -all_plant.CalcInverseDynamics(
      robot_and_ball_context, np.zeros([len(v)]), link_wrenches)

    # Get the robot actuation matrix.
    B = self.controller.ConstructRobotActuationMatrix()

    # Integrate the velocity forward in time.
    dt = 1e-10
    vdot = np.linalg.solve(M, fext + f_contact + B.dot(f_act))
    vnew = v + dt * vdot

    # Get the velocity at the point of contacts.
    Nv = N.dot(vnew)
    Sv = S.dot(vnew)
    Tv = T.dot(vnew)
    self.assertLess(np.linalg.norm(Nv), zero_velocity_tol, msg=dbg_out)
    self.assertLess(np.linalg.norm(Sv), zero_velocity_tol, msg=dbg_out)
    self.assertLess(np.linalg.norm(Tv), zero_velocity_tol, msg=dbg_out)

  # Check control outputs for when contact is intended and robot and ball are
  # indeed in contact and verifies that slip is not caused.
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
        if self.controller.IsRobotContactingBall(contacts):
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
    vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])

    # Get the Jacobians at the point of contact: N, S, T, and construct Z and
    # Zdot_v.
    nc = len(contacts)
    N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q)
    Z = np.zeros([N.shape[0] * 3, N.shape[1]])
    Z[0:nc,:] = N
    Z[nc:2*nc,:] = S
    Z[-nc:,:] = T

    # Set the time-derivatives of the Jacobians times the velocity.
    Zdot_v = np.zeros([nc * 3])
    Zdot_v[0:nc] = Ndot_v[:,0]
    Zdot_v[nc:2*nc] = Sdot_v[:, 0]
    Zdot_v[-nc:] = Tdot_v[:, 0]

    # Determine the control forces using the learned dynamics controller.
    u, fz = self.controller.ComputeContactControlMotorTorquesUsingLearnedDynamics(self.controller_context, M, fext, vdot_ball_des, Z, Zdot_v)

    # Step the plant forward by a small time.
    sim = EmbeddedSim(self.step_size)
    sim.UpdateTime(t)
    sim.UpdatePlantPositions(q)
    sim.UpdatePlantVelocities(v)
    sim.ApplyControls(B.dot(u))
    sim.Step()

    # Compute the approximate acceleration.
    vnew = sim.GetPlantVelocities()
    vdot_approx = np.reshape((vnew - v) / sim.delta_t, (-1, 1))
    vdot_ball_approx = all_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)

    # Get the desired acceleration for the ball.
    nv_ball = self.controller.nv_ball()
    vdot_ball_des = plan.GetBallQVAndVdot(t)[-nv_ball:]

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
    assert self.fully_actuated

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
        if self.controller.IsRobotContactingBall(contacts):
          logging.info('-- TestContactAndContactIntendedOutputsAccelerationCorrect() - desired time identified: ' + str(t))
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
    vdot_ball_des = plan.GetBallQVAndVdot(t)[-nv_ball:]

    # Check the accelerations.
    self.assertAlmostEqual(np.linalg.norm(vdot_ball_approx - vdot_ball_des), 0, places=5)

  # Check control outputs for when robot is not in contact with the ball but it
  # is desired to be.
  def test_NoContactButContactIntendedOutputsCorrect(self):
    # Get the plan.
    plan = self.controller.plan

    # Advance time, finding a point at which contact is desired *and* where
    # the robot is not contacting the ground.
    dbg_out = '\n\nDebugging output follows:'
    dt = 1e-3
    t = 0.0
    t_final = plan.end_time()
    while True:
      if plan.IsContactDesired(t):
        # Look for contact.
        q, v = self.SetStates(t)
        contacts = self.controller.FindContacts(q)
        if not self.controller.IsRobotContactingBall(contacts):
          dbg_out += '\n  -- TestNoContactButContactIntendedOutputsCorrect() - desired time identified: ' + str(t)
          break

      # No contact desired or contact was found.
      t += dt
      if t >= t_final:
        dbg_out += '\n -- TestNoContactButContactIntendedOutputsCorrect() - contact always found!'
        return

    # Use the controller output to determine the generalized acceleration of the robot.
    q_robot = self.controller.get_q_robot(self.controller_context)
    v_robot = self.controller.get_v_robot(self.controller_context)
    robot_context = self.robot_plant.CreateDefaultContext()
    self.robot_plant.SetPositions(robot_context, q_robot)
    self.robot_plant.SetVelocities(robot_context, v_robot)
    M = self.robot_plant.CalcMassMatrixViaInverseDynamics(robot_context)
    link_wrenches = MultibodyForces(self.robot_plant)
    fext = -self.robot_plant.CalcInverseDynamics(
        robot_context, np.zeros([self.controller.nv_robot()]), link_wrenches)
    u_robot = self.controller.robot_and_ball_plant.GetVelocitiesFromArray(self.robot_instance, self.output.get_vector_data(0).CopyToVector())
    vdot_robot = np.linalg.inv(M).dot(u_robot + fext)
    dbg_out += '\nDesired robot velocity: ' + str(vdot_robot)

    # Get the current distance from the robot to the ball.
    old_dist = self.controller.GetSignedDistanceFromRobotToBall(self.controller_context)

    # "Simulate" a small change to the box by computing a first-order
    # discretization to the new velocity and then using that to compute a first-order
    # discretization to the new configuration.
    dt = 1e-3
    vnew_robot = v_robot + dt*vdot_robot
    qd_robot = self.robot_plant.MapVelocityToQDot(robot_context, vnew_robot)
    qnew_robot = q_robot + dt*qd_robot

    # Set the new position and velocity "estimates" of the robot.
    robot_q_input = BasicVector(self.controller.nq_robot())
    robot_v_input = BasicVector(self.controller.nv_robot())
    robot_q_input_vec = robot_q_input.get_mutable_value()
    robot_v_input_vec = robot_v_input.get_mutable_value()
    robot_q_input_vec[:] = qnew_robot
    robot_v_input_vec[:] = vnew_robot
    self.controller_context.FixInputPort(
        self.controller.get_input_port_estimated_robot_q().get_index(),
        robot_q_input)
    self.controller_context.FixInputPort(
        self.controller.get_input_port_estimated_robot_v().get_index(),
        robot_v_input)

    # Get the new distance from the box to the ball.
    new_dist = self.controller.GetSignedDistanceFromRobotToBall(self.controller_context)
    dbg_out += '\nOld distance: ' + str(old_dist) + ' new distance: ' +  str(new_dist)

    self.assertLess(new_dist, old_dist, msg=dbg_out)

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
        if self.controller.IsRobotContactingBall(contacts):
          break

      # No contact desired or contact between robot and ball not found.
      t += dt
      assert t < t_final

    # Setup debugging output.
    dbg_out = '\nDebugging log follows: '
    dbg_out += '\nRobot velocity: ' + str(self.all_plant.GetVelocitiesFromArray(self.controller.robot_instance, v)) + '\n'
    dbg_out += '\nBall velocity: ' + str(self.all_plant.GetVelocitiesFromArray(self.controller.ball_instance, v)) + '\n'
    #v = self.all_plant.SetVelocitiesInArray(self.controller.robot_instance, [0, 0, 1, 0, 0, 0], v)
    #v = self.all_plant.SetVelocitiesInArray(self.controller.ball_instance, [0, 0, 0, 0, 0, 0], v)

    # Construct the Jacobian matrices using the controller function.
    [N, S, T, Ndot, Sdot, Tdot] = self.controller.ConstructJacobians(contacts, q)

    self.PrintContacts(t)
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
      dbg_out += "\nProcessing contact between " + body_A.name() + " and " + body_B.name()

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
      dbg_out += '\nWorld contact point on A: ' + str(point_pair.p_WCa) + '  on B: ' + str(point_pair.p_WCb)
      dbg_out += '\nBody contact point on A: ' + str(p_A) + '  on B: ' + str(p_B)
      dbg_out += '\nq_robot (old): ' + str(self.all_plant.GetPositionsFromArray(self.controller.robot_instance, q))
      dbg_out += '\nq_robot (new): ' + str(self.all_plant.GetPositionsFromArray(self.controller.robot_instance, qnew))
      dbg_out += '\nqdot_robot: ' + str(self.all_plant.GetPositionsFromArray(self.controller.robot_instance, qdot))
      dbg_out += '\nq_ball (old): ' + str(self.all_plant.GetPositionsFromArray(self.controller.ball_instance, q))
      dbg_out += '\nq_ball (new): ' + str(self.all_plant.GetPositionsFromArray(self.controller.ball_instance, qnew))
      dbg_out += '\nqdot_ball: ' + str(self.all_plant.GetPositionsFromArray(self.controller.ball_instance, qdot))
      self.all_plant.SetPositions(robot_and_ball_context, qnew)

      # Determine the new locations of the points on the bodies. The difference
      # in the points yields a finite difference approximation to the relative
      # velocity.
      X_WA_new = self.all_plant.EvalBodyPoseInWorld(robot_and_ball_context, body_A)
      X_WB_new = self.all_plant.EvalBodyPoseInWorld(robot_and_ball_context, body_B)

      # The *velocity* at a point of contact, C, measured in the global frame is
      # the limit as h -> 0 of the difference in point location between t and
      # t + h, divided by h.
      dbg_out += '\nNew point location p_W_A: ' + str(X_WA_new.multiply(p_A))
      dbg_out += '\nNew point location p_W_B: ' + str(X_WB_new.multiply(p_B))
      pdot_W_A_approx = (X_WA_new.multiply(p_A) - X_WA.multiply(p_A)) / dt
      pdot_W_B_approx = (X_WB_new.multiply(p_B) - X_WB.multiply(p_B)) / dt
      dbg_out += '\npdot_W_A (approx): ' + str(pdot_W_A_approx)
      dbg_out += '\npdot_W_B (approx): ' + str(pdot_W_B_approx)
      pdot_W_approx = pdot_W_A_approx - pdot_W_B_approx
      dbg_out += '\npdot_W (approx): ' + str(pdot_W_approx) + '\n'

      # The Jacobian-determined contact point and the new contact point should
      # differ little.
      dbg_out += '\npdot_W (true): ' + str(pdot_W) + '\n'
      self.assertLess(np.linalg.norm(pdot_W.flatten() - pdot_W_approx.flatten()), dt, msg='pdot - ~approx too large (>' + str(dt) + ')' + dbg_out)

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
        N, S, T, Ndot, Sdot, Tdot = self.controller.ConstructJacobians(contacts, q)

        # Output all contacting bodies
        self.PrintContacts(t)

        # Verify that the velocity at the contact point is approximately zero.
        zero_velocity_tol = 1e-12
        Nv = N.dot(v)
        dbg_out = '\n\nDebugging output follows:'
        dbg_out += '\nNv: ' + str(Nv)
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
        N, S, T, Ndot, Sdot, Tdot = self.controller.ConstructJacobians(contacts, q)

        # Output all contacting bodies
        self.PrintContacts(t)

        # Verify that the velocity at the contact points are approximately zero.
        zero_velocity_tol = 1e-10
        Nv = N.dot(v)
        Sv = S.dot(v)
        Tv = T.dot(v)
        dbg_out = '\n\nDebugging output follows:'
        dbg_out += '\nNv: ' + str(Nv)
        dbg_out += '\nSv: ' + str(Sv)
        dbg_out += '\nTv: ' + str(Tv)
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
      dist_robot_ball = self.controller.GetSignedDistanceFromRobotToBall(self.controller_context)
      dist_ground_ball = self.controller.GetSignedDistanceFromBallToGround(self.controller_context)
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
  parser.add_argument("--fully_actuated", type=str2bool, default=False)
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

