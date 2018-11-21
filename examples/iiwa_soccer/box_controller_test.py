import math
import numpy as np
from manipulation_plan import ManipulationPlan
from box_controller import BoxController
from box_soccer_simulation import BuildBlockDiagram

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType,
                         BasicVector, MultibodyForces, ComputeBasisFromAxis)

class ControllerTest:
  def __init__(self, time_step):
    # Construct some reasonable defaults.
    # Gains in Cartesian-land.
    robot_cart_kp = np.ones([3]) * 100
    robot_cart_kd = np.ones([3]) * 10

    # Joint gains for the robot.
    nv_robot = 6
    robot_gv_kp = np.ones([nv_robot]) * 10
    robot_gv_ki = np.ones([nv_robot]) * 0.1
    robot_gv_kd = np.ones([nv_robot]) * 1.0

    self.controller, self.diagram, self.all_plant, self.robot_plant, self.mbw, self.robot_instance, self.ball_instance = BuildBlockDiagram(time_step, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd)

    # Create the context for the diagram.
    self.context = self.diagram.CreateDefaultContext()
    self.controller_context = self.diagram.GetMutableSubsystemContext(self.controller, self.context)
    self.output = self.controller.AllocateOutput()

  # Sets the states for the robot and the ball to those planned.
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
    tree = self.controller.robot_and_ball_plant.tree()
    q = tree.SetPositionsInArray(self.robot_instance, q_robot_des, q)
    q = tree.SetPositionsInArray(self.ball_instance, q_ball_des, q)
    v = tree.SetVelocitiesInArray(self.robot_instance, v_robot_des, v)
    v = tree.SetVelocitiesInArray(self.ball_instance, v_ball_des, v)

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

  # This tests the control matrix for the robot (box).
  def TestControlMatrix(self):
    # Get the control matrix.
    B = self.controller.ConstructRobotActuationMatrix()

    # Set a recognizable velocity vector.
    nv = self.controller.nv_robot() + self.controller.nv_ball()
    v = np.zeros([nv])
    v[:] = np.linspace(1, 10, nv)

    # Copy the elements corresponding to the robot out of v.
    vselected = self.all_plant.tree().GetVelocitiesFromArray(self.controller.robot_instance, v)
    v_all_but_zero = np.zeros([nv])
    v_all_but_zero = self.all_plant.tree().SetVelocitiesInArray(self.controller.robot_instance, vselected, v_all_but_zero)
    vselected_row = np.zeros([len(vselected), 1])
    vselected_row[:,0] = vselected

    # Check whether B*v = v_all_but_zero.
    assert np.allclose(v_all_but_zero, B.dot(vselected).flatten())
    print 'TestControlMatrix() passed'

  # This tests the velocity weighting matrix for the ball.
  def TestBallVelocityMatrix(self):
    # Get the state weighting matrix.
    W = self.controller.ConstructBallVelocityWeightingMatrix()

    # Set a recognizable velocity vector.
    nv = self.controller.nv_robot() + self.controller.nv_ball()
    v = np.zeros([nv])
    v[:] = np.linspace(1, 10, nv)

    # Copy the elements corresponding to the ball out of v.
    vselected = self.all_plant.tree().GetVelocitiesFromArray(self.controller.ball_instance, v)
    v_all_but_zero = np.zeros([nv])
    v_all_but_zero = self.all_plant.tree().SetVelocitiesInArray(self.controller.ball_instance, vselected, v_all_but_zero)
    vselected_row = np.zeros([len(vselected), 1])
    vselected_row[:,0] = vselected

    # Check whether B*v = v_all_but_zero.
    assert np.allclose(v_all_but_zero, B.dot(vselected).flatten())
    print 'TestBallVelocityMatrix() passed'

  # This tests only that q and v for the ball and the robot (box) have
  # expected values.
  def TestQAndVSetProperly(self):
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
    assert np.allclose(q_robot, q_robot_des.flatten())
    assert np.allclose(v_robot, v_robot_des.flatten())
    assert np.allclose(q_ball, q_ball_des.flatten())
    assert np.allclose(v_ball, v_ball_des.flatten())

    # Verify that q is set in the proper place in the array.
    q_all = self.controller.get_q_all(self.controller_context)
    assert np.allclose(q_robot, self.controller.robot_and_ball_plant.tree().GetPositionsFromArray(self.controller.robot_instance, q_all))
    assert np.allclose(q_ball, self.controller.robot_and_ball_plant.tree().GetPositionsFromArray(self.controller.ball_instance, q_all))
    print 'TestDefaultQAndV() passed'

  def TestBodyAccessors(self):
    # This is just a smoke test.
    self.controller.get_ball_from_robot_and_ball_plant()
    self.controller.get_foot_links_from_robot_and_ball_plant()
    self.controller.get_ground_from_robot_and_ball_plant()
    print 'TestBodyAccessors() passed'

  def TestIntegralValue(self):
    int_vec = np.linspace(0, 1, self.controller.nq_robot())
    self.controller.set_integral_value(self.controller_context, int_vec)
    assert np.allclose(self.controller.get_integral_value(self.controller_context), int_vec)
    print 'TestIntegralValue() passed'

  def TestStateSizes(self):
    assert self.controller.nq_ball() == 7
    assert self.controller.nv_ball() == 6
    assert self.controller.nq_robot() == 7
    assert self.controller.nv_robot() == 6
    print 'TestStateSizes() passed'

  # Check control outputs for when contact is intended and robot and ball are
  # indeed in contact.
  def TestContactAndContactIntendedOutputsCorrect(self):
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
          print '  -- TestContactAndContactIntendedOutputsCorrect() - desired time identified: ' + str(t)
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
    assert np.linalg.norm(Nv) < zero_velocity_tol
    assert np.linalg.norm(Sv) < zero_velocity_tol
    assert np.linalg.norm(Tv) < zero_velocity_tol

    # Compute the output from the controller.
    self.controller.CalcOutput(self.controller_context, self.output)

    # Determine the predicted forces due to contact.
    f_act, f_contact = self.controller.ComputeActuationForContactDesiredAndContacting(self.controller_context, contacts)

    # Use the controller output to determine the generalized acceleration of the
    # robot and the ball.
    M = all_plant.tree().CalcMassMatrixViaInverseDynamics(robot_and_ball_context)
    link_wrenches = MultibodyForces(self.all_plant.tree())
    fext = -all_plant.tree().CalcInverseDynamics(
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
    assert np.linalg.norm(Nv) < zero_velocity_tol
    assert np.linalg.norm(Sv) < zero_velocity_tol
    assert np.linalg.norm(Tv) < zero_velocity_tol

    print 'TestContactAndContactIntendedOutputsCorrect() passed'


  # Check control outputs for when robot is not in contact with the ball but it
  # is desired to be.
  def TestNoContactButContactIntendedOutputsCorrect(self):
    # Get the plan.
    plan = self.controller.plan

    # Advance time, finding a point at which contact is desired *and* where
    # the robot is not contacting the ground.
    dt = 1e-3
    t = 0.0
    t_final = plan.end_time()
    while True:
      if plan.IsContactDesired(t):
        # Look for contact.
        q, v = self.SetStates(t)
        contacts = self.controller.FindContacts(q)
        if not self.controller.IsRobotContactingBall(contacts):
          print '  -- TestNoContactButContactIntendedOutputsCorrect() - desired time identified: ' + str(t)
          break

      # No contact desired or contact was found.
      t += dt
      assert t <= t_final

    # Compute the output from the controller.
    self.controller.CalcOutput(self.controller_context, self.output)

    # Use the controller output to determine the generalized acceleration of the robot.
    q_robot = self.controller.get_q_robot(self.controller_context)
    v_robot = self.controller.get_v_robot(self.controller_context)
    robot_context = self.robot_plant.CreateDefaultContext()
    self.robot_plant.SetPositions(robot_context, q_robot)
    self.robot_plant.SetVelocities(robot_context, v_robot)
    M = self.robot_plant.tree().CalcMassMatrixViaInverseDynamics(robot_context)
    link_wrenches = MultibodyForces(self.robot_plant.tree())
    fext = -self.robot_plant.tree().CalcInverseDynamics(
        robot_context, np.zeros([self.controller.nv_robot()]), link_wrenches)
    vdot_robot = np.linalg.inv(M).dot(self.output.get_vector_data(0).CopyToVector() + fext)

    # Get the current distance from the robot to the ball.
    old_dist = self.controller.GetSignedDistanceFromRobotToBall(self.controller_context)

    # "Simulate" a small change to the box by computing a first-order
    # discretization to the new velocity and then using that to compute a first-order
    # discretization to the new configuration.
    dt = 1e-3
    vnew_robot = v_robot + dt*vdot_robot
    qd_robot = self.robot_plant.MapVelocityToQDot(robot_context, vnew_robot, len(q_robot))
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
    print new_dist
    print old_dist
    assert new_dist < old_dist
    print 'TestNoContactButContactIntendedOutputsCorrect() passed'


  # Check that Jacobian construction is correct.
  def TestJacobianConstruction(self):
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

      # No contact desired or exactly one contact not found.
      t += dt
      assert t < t_final

    v = self.all_plant.tree().SetVelocitiesInArray(self.controller.robot_instance, [0, 0, 1, 0, 0, 0], v)
    v = self.all_plant.tree().SetVelocitiesInArray(self.controller.ball_instance, [0, 0, 0, 0, 0, 0], v)

    # Construct the Jacobian matrices using the controller function.
    [N, S, T, Ndot, Sdot, Tdot] = self.controller.ConstructJacobians(contacts, q)
    print 'N: '
    print N

    # Set a time step.
    dt = 1e-6

    # Get a single contact.
    point_pair = contacts[0]

    # Get the two bodies.
    robot_and_ball_context = self.controller.robot_and_ball_context
    self.all_plant.SetPositions(robot_and_ball_context, q)
    query_object = self.all_plant.EvalAbstractInput(
        robot_and_ball_context, self.controller.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()
    geometry_A_id = point_pair.id_A
    geometry_B_id = point_pair.id_B
    frame_A_id = inspector.GetFrameId(geometry_A_id)
    frame_B_id = inspector.GetFrameId(geometry_B_id)
    body_A = self.all_plant.GetBodyFromFrameId(frame_A_id)
    body_B = self.all_plant.GetBodyFromFrameId(frame_B_id)
    print body_A.name()
    print body_B.name()

    # The Jacobians yield the instantaneous movement of a contact point along
    # the various directions. Determine the location of the contact point
    # dt into the future in the world frame.
    kXAxisIndex = 0
    R_WC = ComputeBasisFromAxis(kXAxisIndex, point_pair.nhat_BA_W)
    pdot_C = np.zeros([3, 1])
    pdot_C[0] = N.dot(v)
    pdot_C[1] = S.dot(v)
    pdot_C[2] = T.dot(v)
    pdot_W = R_WC.dot(pdot_C)

    # Get the components of the contact point in the body frames.
    X_WA = self.all_plant.tree().EvalBodyPoseInWorld(robot_and_ball_context, body_A)
    X_WB = self.all_plant.tree().EvalBodyPoseInWorld(robot_and_ball_context, body_B)
    p_A = X_WA.inverse().multiply(point_pair.p_WCa)
    p_B = X_WB.inverse().multiply(point_pair.p_WCb)

    # Update the generalized coordinates of the multibodies using a first-order
    # approximation and the arbitrary velocity.
    qdot = self.all_plant.MapVelocityToQDot(robot_and_ball_context, v, len(q))
    qnew = q + dt*qdot
    q_ball = self.all_plant.tree().GetPositionsFromArray(self.controller.ball_instance, qnew)
    print 'q_robot: ' + str(self.all_plant.tree().GetPositionsFromArray(self.controller.robot_instance, qnew))
    print 'q_ball: ' + str(q_ball)
    self.all_plant.SetPositions(robot_and_ball_context, qnew)

    # Determine the new locations of the points on the bodies. The difference
    # in the points yields a finite difference approximation to the relative
    # velocity.
    X_WA_new = self.all_plant.tree().EvalBodyPoseInWorld(robot_and_ball_context, body_A)
    X_WB_new = self.all_plant.tree().EvalBodyPoseInWorld(robot_and_ball_context, body_B)
    pdot_W_approx = (X_WA_new.multiply(p_A) - X_WB_new.multiply(p_B))/dt

    # The Jacobian-determined contact point and the new contact point should
    # differ little.
    print 'pdot_W (true): ' + str(pdot_W)
    print 'pdot_W (approx): ' + str(pdot_W_approx)
    assert np.linalg.norm(pdot_W.flatten() - pdot_W_approx.flatten()) < dt
    print 'TestJacobianConstruction() passed'

  # Check that velocity at the contact point remains sufficiently close to zero.
  def TestZeroVelocityAtContact(self):
    # Get the plan.
    plan = self.controller.plan

    # Advance time, finding a point at which contact is desired *and* where the
    # robot is contacting the ball.
    dt = 1e-3
    t = 0.0
    t_final = plan.end_time()
    while True:
      # Set the planned q and v.
      if plan.IsContactDesired(t):
        q, v = self.SetStates(t)

      # Look for two contacts: robot/ball and ball/ground.
      contacts = self.controller.FindContacts(q)
      if self.controller.IsRobotContactingBall(contacts):
        break

      # No contact desired or exactly two contacts not found.
      t += dt
      assert t < t_final

    # Get the *planned* point of contact in the world.
    # Set the contact point to an arbitrary point in the world.
    p = plan.GetContactKinematics(t)[0:3]

    # Get the computed contact point location in the world.
    pr_WA = contacts[0].p_WCa
    pr_WB = contacts[0].p_WCb
    p_true = (pr_WA + pr_WB) * 0.5

    # Verify that the planned and actual contact points are collocated.
    assert np.linalg.norm(p.flatten() - p_true.flatten()) < 1e-8

    # Construct the Jacobian matrices using the controller function.
    N, S, T, Ndot, Sdot, Tdot = self.controller.ConstructJacobians(contacts, q)

    # Verify that the velocity at the contact points are approximately zero.
    zero_velocity_tol = 1e-12
    Nv = N.dot(v)
    Sv = S.dot(v)
    Tv = T.dot(v)
    print 'Nv: ' + str(Nv)
    print 'Sv: ' + str(Sv)
    print 'Tv: ' + str(Tv)
    assert np.linalg.norm(Nv) < zero_velocity_tol
    assert np.linalg.norm(Sv) < zero_velocity_tol
    assert np.linalg.norm(Tv) < zero_velocity_tol
    print 'TestZeroVelocityAtContact() passed'


  # Check that the contact distance when the plan indicates contact is desired
  # always lies below a threshold.
  def TestContactDistanceBelowThreshold(self):
    # Get the plan.
    plan = self.controller.plan

    # Advance time, finding a point at which contact is desired.
    dt = 1e-3
    t = 0.0
    while True:
      # Determine whether the plan indicates contact is desired.
      if not plan.IsContactDesired(t):
        t += dt
        continue

      # Set the states to q and v.
      q, v = self.SetStates(t)

      # Check the distance between the robot foot and the ball.
      dist_thresh = 1e-6
      assert abs(self.controller.GetSignedDistanceFromRobotToBall(self.controller_context)) < dist_thresh
      assert abs(self.controller.GetSignedDistanceFromBallToGround(self.controller_context)) < dist_thresh
      assert False

      # Update t.
      t += dt

    print 'TestContactDistanceBelowThreshold() passed'


def main():
  test = ControllerTest(0.0)
  test.TestStateSizes()
  test.TestIntegralValue()
  test.TestBodyAccessors()
  test.TestQAndVSetProperly()
  test.TestControlMatrix()
  # test.TestBallVelocityMatrix()
  test.TestJacobianConstruction()
  test.TestContactAndContactIntendedOutputsCorrect()
  test.TestNoContactButContactIntendedOutputsCorrect()
  test.TestContactDistanceBelowThreshold()
  test.TestZeroVelocityAtContact()


if __name__ == "__main__":
  main()
  