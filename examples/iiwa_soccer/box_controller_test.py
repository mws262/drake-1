import math
import sys
import numpy as np
import unittest
import argparse
import logging
import scipy.optimize
import cma

from manipulation_plan import ManipulationPlan
from box_controller import BoxController
from box_soccer_simulation import BuildBlockDiagram
from embedded_box_soccer_sim import EmbeddedSim

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType, BasicVector, MultibodyForces,
                          ComputeBasisFromAxis, MathematicalProgram)
from pydrake.solvers import mathematicalprogram

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
      logging.info('Contact found between ' + body_A.name() + ' and ' + body_B.name() + ' at: ' + str(i.p_WCa))
      logging.info('  Point of contact: ' + str(0.5 * (i.p_WCa + i.p_WCb)))
      logging.info('  Normal (pointing to ' + body_A.name() + '): ' + str(i.nhat_BA_W))

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

  # Computes the applied forces when the ball is fully actuated (version that
  # simply solves a linear system). This is a function for debugging purposes.
  # It shows that the desired accelerations are feasible using *some* forces
  # (not necessarily ones that are consistent with kinematic/force constraints).
  def ComputeFullyActuatedBallControlForces(self, controller_context):
    assert self.fully_actuated

    # Get the generalized inertia matrix.
    all_plant = self.robot_and_ball_plant
    all_context = self.robot_and_ball_context
    M = all_plant.CalcMassMatrixViaInverseDynamics(all_context)
    iM = np.linalg.inv(M)

    # Compute the contribution from force elements.
    link_wrenches = MultibodyForces(all_plant)
    all_plant.CalcForceElementsContribution(all_context, link_wrenches)

    # Compute the external forces.
    fext = -all_plant.CalcInverseDynamics(all_context, np.zeros([self.nv_robot() + self.nv_ball()]), link_wrenches)
    fext = np.reshape(fext, [-1, 1])

    # Get the desired ball *linear* acceleration.
    vdot_ball_des = self.plan.GetBallQVAndVdot(controller_context.get_time())[-3:]
    vdot_ball_des = np.reshape(vdot_ball_des, [-1, 1])

    # Construct the actuation and weighting matrices.
    B = self.ConstructRobotActuationMatrix()
    P = self.ConstructBallVelocityWeightingMatrix()

    # Primal variables are motor torques and accelerations.
    nv, nu = B.shape
    nprimal = nu + nv

    # Initialize epsilon.
    epsilon = np.zeros([nv, 1])

    # Get the current system positions and velocities.
    q = self.get_q_all(controller_context)
    v = self.get_v_all(controller_context)

    # Solve the equation:
    # M\dot{v} - Bu = fext + epsilon
    # where:
    # \dot{v} = | vdot_ball_des |
    #           | 0             |
    dotv = np.zeros([len(v), 1])
    all_plant.SetVelocitiesInArray(self.ball_instance, vdot_ball_des, dotv)

    # Set maximum number of loop iterations.
    max_loop_iterations = 100

    for i in range(max_loop_iterations):
        # Compute b.
        b = M.dot(dotv) - fext - epsilon
        [z, residuals, rank, singular_values] = np.linalg.lstsq(B, b)
        u = np.reshape(z, [-1])

        # Update the state in the embedded simulation.
        self.embedded_sim.UpdateTime(controller_context.get_time())
        self.embedded_sim.UpdatePlantPositions(q)
        self.embedded_sim.UpdatePlantVelocities(v)

        # Apply the controls to the embedded simulation.
        self.embedded_sim.ApplyControls(B.dot(u))

        # Simulate the system forward in time.
        self.embedded_sim.Step()

        # Get the new system velocity.
        vnew = self.embedded_sim.GetPlantVelocities()

        # Compute the estimated acceleration.
        vdot_approx = np.reshape((vnew - v) / self.embedded_sim.delta_t, (-1, 1))

        # Compute delta-epsilon.
        delta_epsilon = M.dot(vdot_approx) - fext - B.dot(np.reshape(u, (-1, 1))) - epsilon

        # If delta-epsilon is sufficiently small, quit.
        if np.linalg.norm(delta_epsilon) < 1e-6:
            break

        # Update epsilon.
        epsilon += delta_epsilon

    if i == max_loop_iterations - 1:
      logging.warning('BlackBoxDynamics controller did not terminate!')

    z = np.reshape(z, [-1, 1])
    logging.debug('u: ' + str(u))
    logging.debug('objective: ' + str(0.5*np.linalg.norm(P.dot(vdot_approx) - vdot_ball_des)))
    logging.debug('Delta epsilon norm: ' + str(np.linalg.norm(delta_epsilon)))
    logging.debug('Ball acceleration: ' + str(all_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)))
    logging.debug('P * vdot (computed): ' + str(P.dot(z[0:nv])))
    logging.debug('P * vdot (approx): ' + str(P.dot(vdot_approx)))

    return u

  # Computes the motor torques for ComputeActuationForContactDesiredAndContacting() that minimize deviation from the
  # desired acceleration using no dynamics information and a gradient-free optimization strategy.
  # This controller uses the simulator to compute contact forces, rather than attempting to predict the contact forces
  # that the simulator will generate.
  def ComputeOptimalContactControlMotorTorquesDerivativeFree(self, controller_context, vdot_ball_des):

    P = self.controller.ConstructBallVelocityWeightingMatrix()
    nu = self.controller.ConstructRobotActuationMatrix().shape[1]

    # Get the current system positions and velocities.
    q = self.controller.get_q_all(controller_context)
    v = self.controller.get_v_all(controller_context)
    nv = len(v)

    # The objective function.
    def objective_function(u):
      vdot_approx = self.controller.ComputeApproximateAcceleration(controller_context, q, v, u)
      delta = P.dot(vdot_approx) - vdot_ball_des
      return np.linalg.norm(delta)

    # Do CMA-ES.
    sigma = 100.0
    u_best = np.zeros(nu)
    fbest = objective_function(u_best)
    for i in range(1):
      es = cma.CMAEvolutionStrategy(np.random.randn(nu) * 100, sigma)
      es.optimize(objective_function)
      if es.result.fbest < fbest:
        fbest = es.result.fbest
        u_best = es.result.xbest

    return u_best

  # Computes the applied forces when the ball is fully actuated (version that
  # solves a QP). This is a function for debugging purposes. It shows that the
  # optimization criterion that we use is a reasonable one; see the unit test
  # for this function.
  def ComputeFullyActuatedBallControlForces(self, controller_context):
    assert self.fully_actuated

    # Get the generalized inertia matrix.
    all_plant = self.robot_and_ball_plant
    all_context = self.robot_and_ball_context
    M = all_plant.CalcMassMatrixViaInverseDynamics(all_context)
    iM = np.linalg.inv(M)

    # Compute the contribution from force elements.
    link_wrenches = MultibodyForces(all_plant)
    all_plant.CalcForceElementsContribution(all_context, link_wrenches)

    # Compute the external forces.
    fext = -all_plant.CalcInverseDynamics(all_context, np.zeros([self.nv_robot() + self.nv_ball()]), link_wrenches)
    fext = np.reshape(fext, [-1, 1])

    # Get the desired ball acceleration.
    vdot_ball_des = self.plan.GetBallQVAndVdot(controller_context.get_time())[-self.nv_ball():]
    vdot_ball_des = np.reshape(vdot_ball_des, [-1, 1])

    # Construct the weighting matrix. This must be done manually so that we
    # capture *all* degrees-of-freedom.
    nv = self.nv_ball() + self.nv_robot()
    v = np.zeros([nv])
    dummy_ball_v = np.array([1, 1, 1, 1, 1, 1])
    self.robot_and_ball_plant.SetVelocitiesInArray(self.ball_instance, dummy_ball_v, v)
    P = np.zeros([self.nv_ball(), nv])
    vball_index = 0
    for i in range(nv):
      if abs(v[i]) > 0.5:
        P[vball_index, i] = v[i]
        vball_index += 1

    # Primal variables are motor torques and accelerations.
    nu = nv
    nprimal = nu + nv

    # Construct the Hessian matrix and linear term.
    # min (P*vdot_new - vdot_des_ball)^2
    H = np.zeros([nprimal, nprimal])
    H[0:nv,0:nv] = P.T.dot(P) + np.eye(nv) * 1e-6
    c = np.zeros([nprimal, 1])
    c[0:nv] = -P.T.dot(vdot_ball_des)

    # Construct the equality constraint matrix (for the QP) corresponding to:
    # M\dot{v} - Bu = fext + epsilon and

    A = np.zeros([nv, nv + nu])
    A[0:nv,0:nv] = M
    A[0:nv,nv:] = -np.eye(nv)

    # Initialize epsilon.
    epsilon = np.zeros([nv, 1])

    # Get the current system positions and velocities.
    q = self.get_q_all(controller_context)
    v = self.get_v_all(controller_context)

    # Set maximum number of loop iterations.
    max_loop_iterations = 100

    for i in range(max_loop_iterations):
        # Compute b.
        b = np.zeros([nv, 1])
        b[0:nv] = fext + epsilon
        vdot_des = np.zeros([nv, 1])
        all_plant.SetVelocitiesInArray(self.ball_instance, vdot_ball_des, vdot_des)

        # Solve the QP.
        prog = mathematicalprogram.MathematicalProgram()
        vars = prog.NewContinuousVariables(len(c), "vars")
        prog.AddQuadraticCost(H, c, vars)
        prog.AddLinearConstraint(A, b, b, vars)
        result = prog.Solve()
        if result != mathematicalprogram.SolutionResult.kSolutionFound:
            print result
            assert False
        z = prog.GetSolution(vars)
        u = np.reshape(z[nv:], [-1, 1])
        vdot = np.reshape(z[0:nv], [-1, 1])

        # Update the state in the embedded simulation.
        self.embedded_sim.UpdateTime(controller_context.get_time())
        self.embedded_sim.UpdatePlantPositions(q)
        self.embedded_sim.UpdatePlantVelocities(v)

        # Apply the controls to the embedded simulation.
        self.embedded_sim.ApplyControls(u)

        # Simulate the system forward in time.
        self.embedded_sim.Step()

        # Get the new system velocity.
        vnew = self.embedded_sim.GetPlantVelocities()

        # Compute the estimated acceleration.
        vdot_approx = np.reshape((vnew - v) / self.embedded_sim.delta_t, (-1, 1))

        # Compute delta-epsilon.
        delta_epsilon = M.dot(vdot_approx) - fext - np.reshape(u, (-1, 1)) - epsilon

        # If delta-epsilon is sufficiently small, quit.
        if np.linalg.norm(delta_epsilon) < 1e-6:
            break

        # Update epsilon.
        epsilon += delta_epsilon

    if i == max_loop_iterations - 1:
      logging.warning('BlackBoxDynamics controller did not terminate!')

    # We know that M\dot{v}* = fext + u
    # and we know that M\dot{v} - u = fext

    self.ball_accel_from_controller = all_plant.GetVelocitiesFromArray(self.ball_instance, z[0:nv])[-3:]
    z = np.reshape(z, [-1, 1])
    logging.debug('z: ' + str(z))
    logging.debug('u: ' + str(u))
    logging.debug('Delta epsilon norm: ' + str(np.linalg.norm(delta_epsilon)))
    logging.debug('Ball acceleration: ' + str(all_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)))
    logging.debug('objective (opt): ' + str(z.T.dot(0.5 * H.dot(z) + c)))
    logging.debug('objective (true): ' + str(z.T.dot(0.5 * H.dot(z) + c) + 0.5*vdot_ball_des.T.dot(vdot_ball_des)))
    logging.debug('vdot_approx - vdot (computed): ' + str(vdot_approx - z[0:nv]))
    logging.debug('A * z - b: ' + str(A.dot(z) - b))
    logging.debug('P * vdot (computed): ' + str(P.dot(z[0:nv])))
    logging.debug('P * vdot (approx): ' + str(P.dot(vdot_approx)))

    return u

  # Function for constructing Z and Zdot_v from N, S, T, etc.
  def SetZAndZdot(self, N, S, T, Ndot_v, Sdot_v, Tdot_v):
    nc = N.shape[0]

    # Set Z and Zdot_v
    Z = np.zeros([N.shape[0] * 3, N.shape[1]])
    Z[0:nc,:] = N
    Z[nc:2*nc,:] = S
    Z[-nc:,:] = T
    Zdot_v = np.zeros([nc * 3])
    Zdot_v[0:nc] = Ndot_v[:,0]
    Zdot_v[nc:2*nc] = Sdot_v[:, 0]
    Zdot_v[-nc:] = Tdot_v[:, 0]

    return [Z, Zdot_v]

  # Computes the contact forces under the no-slip solution *without also trying to minimize the deviation from the
  # desired acceleration.*
  def ComputeContactForcesWithoutControl(self, q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v):
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
      # fz = argmin 1/2 fn' * (N * vdot - S_vdot)
      # subject to:          M * vdot = f + N' * fn + S' * fs + T' * ft
      #             N * vdot - S_vdot >= 0
      #             S * vdot - S_vdot = 0
      #             T * vdot - T_vdot = 0
      # fn >= 0
      #
      # This problem is only solved if the objective function is zero. It
      # *should* always be solvable.
      #
      # The objective function of the above is:
      # 1/2 fn' * N * (inv(M) * (f + Z'*fz)) + Ndot_v) =
      #    fn' * N * inv(M) * Z' * fz + fn' * (N * inv(M) * f + Ndot_v)
      #
      # The Hessian matrix is: 1/2 * [N; 0] * inv(M) * Z'
      # The gradient is: ([N; 0] * inv(M) * f + [Ndot_v; 0])

      # Form the 'Z' matrix.
      [Z, Zdot_v] = self.SetZAndZdot(N, S, T, Ndot_v, Sdot_v, Tdot_v)

      # Augment the N terms.
      nc = N.shape[0]
      nv = N.shape[1]
      N_aug = np.zeros([nc*3, nv])
      N_aug[0:nc, :] + N
      Ndot_v_aug = np.zeros([nc * 3, 1])    

      # Compute the Hessian.
      H = N_aug.dot(iM.dot(Z.T)) 

      # Compute the linear term.
      c = N_aug.dot(iM.dot(fext)) + Ndot_v_aug

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

      # N*vdot + Ndot_v = 0 ==> N(inv(M) * (f + Z'*fz)) + Ndot_v = 0, etc.
      # S*vdot + Sdot_v = 0 ==> S(inv(M) * (f + Z'*fz)) + Sdot_v = 0, etc.
      # T*vdot + Sdot_v = 0 ==> S(inv(M) * (f + Z'*fz)) + Sdot_v = 0, etc.
      rhs_n = -N.dot(iM).dot(fext) - Ndot_v
      prog.AddLinearConstraint(N.dot(iM).dot(Z.T), rhs_n, rhs_n, vars)
      rhs_s = -S.dot(iM).dot(fext) - Sdot_v
      prog.AddLinearConstraint(S.dot(iM).dot(Z.T), rhs_s, rhs_s, vars)
      rhs_t = -T.dot(iM).dot(fext) - Tdot_v
      prog.AddLinearConstraint(T.dot(iM).dot(Z.T), rhs_t, rhs_t, vars)

      # Solve the QP.
      result = prog.Solve()
      self.assertEquals(result, mathematicalprogram.SolutionResult.kSolutionFound, msg='Unable to solve QP, reason: ' + str(result))
      fz = np.reshape(prog.GetSolution(vars), [-1, 1])
      return [fz, Z, iM, fext]


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
      [Z, Zdot_v] = self.SetZAndZdot(N, S, T, Ndot_v, Sdot_v, Tdot_v)

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

  # Checks that the ball can be accelerated while maintaining the no-slip condition (i.e., and assuming that the state
  # of the ball does not correspond to slipping). We check this by seeing whether contact forces exist that can realize
  # the desired acceleration on the ball.
  def test_NoSlipAcceleration(self):
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

    # Get the plan.
    plan = self.controller.plan

    # Get the robot/ball plant and the correpsonding context from the controller.
    all_plant = self.controller.robot_and_ball_plant
    all_context = self.controller.robot_and_ball_context

    # Set tolerances.
    zero_velocity_tol = 1e-6
    zero_accel_tol = 1e-6
    complementarity_tol = 1e-6

    # TODO: test this with actual contact.
    '''
    # First test a simple known configuration that should pass.
    # 1. Set the state arbitrarily.
    logging.info('About to test a known configuration that should pass.')
    q, v = self.SetStates(0)

    # 2. Hand craft the Jacobians for "point of contact" in the middle of the
    # ball.
    nc = 1
    N = np.zeros([nc, len(v)])
    S = np.zeros([nc, len(v)])
    T = np.zeros([nc, len(v)])
    # Allow ball to be pushed along +x, +y, and +z.
    N[0, 9] = 1
    S[0, 10] = 1
    T[0, 11] = 1
    Ndot_v = np.zeros([nc, 1])
    Sdot_v = Ndot_v
    Tdot_v = Ndot_v
    [Z, Zdot_v] = self.SetZAndZdot(N, S, T, Ndot_v, Sdot_v, Tdot_v)

    # 3. Set the ball acceleration to something that should be easy to achieve.
    vdot_ball_des = np.reshape(np.array([0, 0, 0, 1, 1, 1]), [-1, 1])
    logging.debug('desired ball acceleration (angular/linear): ' + str(vdot_ball_des))

    # 4. Solve the QP.
    [fz, Z, iM, fext] = self.ComputeContactForces(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v, P, vdot_ball_des, check_no_slip=False)
    logging.debug('contact force magnitudes along contact normals: ' + str(fz[0:nc]))
    logging.debug('contact force magnitudes along first contact tangents: ' + str(fz[nc:nc*2]))
    logging.debug('contact force magnitudes along second contact tangents: ' + str(fz[-nc:]))

    # Compute the value at the objective function.
    vdot = iM.dot(fext + Z.T.dot(fz))
    vdot_ball = P.dot(vdot)
    logging.debug('actual ball acceleration: ' + str(vdot_ball))

    # If the objective function value is zero, the acceleration of the ball
    # is realizable without slip.
    fval = np.linalg.norm(vdot_ball - vdot_ball_des)
    self.assertLess(fval, zero_accel_tol)
    '''

    # Advance time, finding a point at which contact is desired *and* where
    # the robot is contacting the ball.
    dt = 1e-3
    t = 0.0
    t_final = plan.end_time()
    while t < t_final:
      # Keep looping if contact is not desired.
      if not plan.IsContactDesired:
        t += dt
        continue

      # Look for contact.
      q, v = self.SetStates(t)
      contacts = self.controller.FindContacts(q)
      if not self.controller.IsRobotContactingBall(contacts):
        t += dt
        continue

      logging.info('-- TestNoSlipAcceleration() - identified testable time/state at t=' + str(t))
      self.PrintContacts(t)

      # Get the desired acceleration.
      vdot_ball_des = plan.GetBallQVAndVdot(self.controller_context.get_time())[-self.controller.nv_ball():]
      vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])[-6:]
      logging.debug('desired ball acceleration (angular/linear): ' + str(vdot_ball_des))

      # Get the Jacobians.
      N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)
      nc = N.shape[0]

      # Verify that the velocities in each direction of the contact frame are zero.
      Nv = N.dot(v)
      Sv = S.dot(v)
      Tv = T.dot(v)
      self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
      self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
      self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

      # Solve the QP without control forces.
      fz_no_control, Z, iM, fext = self.ComputeContactForcesWithoutControl(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v)

      # Solve the QP.
      fz, Z, iM, fext = self.ComputeContactForces(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v, P, vdot_ball_des, enforce_no_slip=False)
      logging.debug('contact force magnitudes along contact normals: ' + str(fz[0:nc]))
      logging.debug('contact force magnitudes along first contact tangents: ' + str(fz[nc:nc*2]))
      logging.debug('contact force magnitudes along second contact tangents: ' + str(fz[-nc:]))

      # Verify that the no-separation controller arrives at the same objective (or better).
      P_star = self.controller.ConstructBallVelocityWeightingMatrix()
      B = self.controller.ConstructRobotActuationMatrix()
      fz_star, Z, iM, fext = self.ComputeContactForces(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v,
          P_star, vdot_ball_des[-3:], enforce_no_slip=False)
      vdot_compute_contact_forces = iM.dot(fext + Z.T.dot(fz_star))
      u, fz_no_separate = self.controller.ComputeContactControlMotorTorquesNoSeparation(iM, fext, vdot_ball_des[-3:], Z, N, Ndot_v)
      vdot_controller = iM.dot(fext + B.dot(u) + Z.T.dot(fz_no_separate))
      self.assertLessEqual(np.linalg.norm(P_star.dot(vdot_controller) - vdot_ball_des[-3:]),
                           np.linalg.norm(P_star.dot(vdot_compute_contact_forces) - vdot_ball_des[-3:]))

      # Compute the acceleration at the optimum.
      vdot = iM.dot(fext + Z.T.dot(fz))
      vdot_ball = P.dot(vdot)
      logging.debug('actual ball acceleration: ' + str(vdot_ball))

      # Get the accelerations of the ball along the contact directions.
      Nvdot = N.dot(vdot) + Ndot_v
      Svdot = S.dot(vdot) + Sdot_v
      Tvdot = T.dot(vdot) + Tdot_v
      logging.debug('acceleration along contact normals: ' + str(Nvdot))
      logging.debug('acceleration along first contact tangents: ' + str(Svdot))
      logging.debug('acceleration along second contact tangents: ' + str(Tvdot))

      # If the objective function value is zero, the acceleration of the ball
      # is realizable without slip.
      vdot_no_control = iM.dot(fext + Z.T.dot(fz_no_control))
      logging.debug('objective function value with NO control forces applied: ' + str(np.linalg.norm(P.dot(vdot_no_control) - vdot_ball_des)))
      fval = np.linalg.norm(vdot_ball - vdot_ball_des)
      self.assertLess(fval, zero_accel_tol)

      # First verify that the no-slip condition *can* be satisfied.
      self.ComputeContactForces(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v, P, vdot_ball_des, enforce_no_slip=True)

      # Verify that the complementarity condition is satisfied.
      nc = N.shape[0]
      self.assertLess(fz[0:nc,0].dot(Nvdot), complementarity_tol,
          msg='Complementarity constraint violated')

      # Verify that the no-slip condition is satisfied.
      self.assertLess(np.linalg.norm(Svdot), zero_accel_tol, msg='No slip constraint violated')
      self.assertLess(np.linalg.norm(Tvdot), zero_accel_tol, msg='No slip constraint violated')

      t += dt

  # Checks that the contact forces computed by the no-separation controller are consistent.
  def test_NoSeparationControllerContactForcesConsistent(self):
    # Get the plan.
    plan = self.controller.plan

    # Get the robot/ball plant and the correpsonding context from the controller.
    all_plant = self.controller.robot_and_ball_plant
    all_context = self.controller.robot_and_ball_context

    # Set tolerances.
    zero_velocity_tol = 1e-6
    force_tol = 1e-6

    # Advance time, finding a point at which contact is desired *and* where
    # the robot is contacting the ball.
    dt = 1e-3
    t = 0.0
    t_final = plan.end_time()
    while t < t_final:
      # Keep looping if contact is not desired.
      if not plan.IsContactDesired:
        t += dt
        continue

      # Look for contact.
      q, v = self.SetStates(t)
      contacts = self.controller.FindContacts(q)
      if not self.controller.IsRobotContactingBall(contacts):
        t += dt
        continue

      logging.info('-- test_NoSeparationControllerContactForcesConsistent() - identified testable time/state at '
                   't=' + str(t))
      self.PrintContacts(t)

      # Get the desired acceleration.
      vdot_ball_des = plan.GetBallQVAndVdot(self.controller_context.get_time())[-self.controller.nv_ball():]
      vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])[-3:]
      logging.debug('desired ball linear acceleration: ' + str(vdot_ball_des))

      # Get the Jacobians.
      N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)
      nc = N.shape[0]

      # Verify that the velocities in each direction of the contact frame are zero.
      Nv = N.dot(v)
      Sv = S.dot(v)
      Tv = T.dot(v)
      self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
      self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
      self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

      # Run the no-slip controller.
      dummy, Z, iM, fext = self.ComputeContactForcesWithoutControl(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v)
      P = self.controller.ConstructBallVelocityWeightingMatrix()
      B = self.controller.ConstructRobotActuationMatrix()
      u, fz = self.controller.ComputeContactControlMotorTorquesNoSeparation(iM, fext, vdot_ball_des, Z, N, Ndot_v)
      vdot = iM.dot(fext + B.dot(u) + Z.T.dot(fz))

      # Get the accelerations of the ball along the directions of the ball contact normals.
      Nvdot = N.dot(vdot) + Ndot_v
      logging.debug('acceleration along contact normals: ' + str(Nvdot))

      # Verify that the contact forces are compressive.
      self.assertGreaterEqual(fz[0:nc].min(), -force_tol)

      # Verify that the normal accelerations are zero.
      self.assertAlmostEqual(Nvdot.min(), 0)
      self.assertAlmostEqual(Nvdot.max(), 0)

      t += dt

  # Check control outputs for when contact is intended and robot and ball are indeed in contact and verifies that slip
  # is not caused *using the embedded simulation*.
  def test_ContactAndContactIntendedOutputsDoNotCauseSlip(self):
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
          logging.info('  -- TestContactAndContactIntendedOutputsCorrect() - desired time identified: ' + str(t))
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
    N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)
    zero_velocity_tol = 1e-3
    Nv = N.dot(v)
    Sv = S.dot(v)
    Tv = T.dot(v)
    self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
    self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
    self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

    # Determine the predicted forces due to contact.
    u = self.controller.ComputeActuationForContactDesiredAndContacting(self.controller_context, contacts)

    # Use the controller output to determine the generalized acceleration of the
    # robot and the ball.
    M = all_plant.CalcMassMatrixViaInverseDynamics(robot_and_ball_context)
    link_wrenches = MultibodyForces(self.all_plant)
    fext = -all_plant.CalcInverseDynamics(
      robot_and_ball_context, np.zeros([len(v)]), link_wrenches)

    # Get the robot actuation matrix.
    B = self.controller.ConstructRobotActuationMatrix()

    # Integrate the velocity forward in time.
    dt = 1e-6
    vdot = np.reshape(self.controller.ComputeApproximateAcceleration(self.controller_context, q, v, u, dt=dt), -1)
    vnew = v + dt * vdot

    # Get the velocity at the point of contacts.
    Nv = N.dot(vnew)
    Sv = S.dot(vnew)
    Tv = T.dot(vnew)
    self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
    self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
    self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

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
    vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])[-3:]

    # Determine the control forces using the learned dynamics controller.
    u = self.ComputeOptimalContactControlMotorTorquesDerivativeFree(self.controller_context, vdot_ball_des)

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
      print 'test_FullyActuatedAccelerationCorrect() only works with --fully_actuated=True option.'
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
        if self.controller.IsRobotContactingBall(contacts):
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

  # Check control outputs for when robot is not in contact with the ball but it
  # is desired to be.
  def test_NoContactButContactIntendedOutputsCorrect(self):
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
          logging.debug('-- TestNoContactButContactIntendedOutputsCorrect() - desired time identified: ' + str(t))
          break

      # No contact desired or contact was found.
      t += dt
      if t >= t_final:
        logging.debug(' -- TestNoContactButContactIntendedOutputsCorrect() - contact always found!')
        return

    # Use the controller output to determine the generalized acceleration of the robot.
    q_robot = self.controller.get_q_robot(self.controller_context)
    v_robot = np.reshape(self.controller.get_v_robot(self.controller_context), [-1, 1])
    robot_context = self.robot_plant.CreateDefaultContext()
    self.robot_plant.SetPositions(robot_context, q_robot)
    self.robot_plant.SetVelocities(robot_context, v_robot)
    M = self.robot_plant.CalcMassMatrixViaInverseDynamics(robot_context)
    link_wrenches = MultibodyForces(self.robot_plant)
    fext = -np.reshape(self.robot_plant.CalcInverseDynamics(
        robot_context, np.zeros([self.controller.nv_robot()]), link_wrenches), [-1, 1])
    u = self.controller.ComputeActuationForContactDesiredButNoContact(self.controller_context)
    #u = self.controller.robot_and_ball_plant.GetVelocitiesFromArray(self.robot_instance, self.output.get_vector_data(0).CopyToVector())
    logging.debug('u: ' + str(u))
    vdot_robot = np.linalg.inv(M).dot(u + fext)
    logging.debug('Desired robot acceleration: ' + str(vdot_robot))

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
    robot_v_input_vec[:] = vnew_robot[:,0]
    self.controller_context.FixInputPort(
        self.controller.get_input_port_estimated_robot_q().get_index(),
        robot_q_input)
    self.controller_context.FixInputPort(
        self.controller.get_input_port_estimated_robot_v().get_index(),
        robot_v_input)

    # Get the new distance from the box to the ball.
    new_dist = self.controller.GetSignedDistanceFromRobotToBall(self.controller_context)
    logging.debug('Old distance: ' + str(old_dist) + ' new distance: ' +  str(new_dist))

    self.assertLess(new_dist, old_dist)

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
    logging.debug('Robot velocity: ' + str(self.all_plant.GetVelocitiesFromArray(self.controller.robot_instance, v)))
    logging.debug('Ball velocity: ' + str(self.all_plant.GetVelocitiesFromArray(self.controller.ball_instance, v)))

    # Construct the Jacobian matrices using the controller function.
    [N, S, T, Ndot_v, Sdot_v, Tdot_v] = self.controller.ConstructJacobians(contacts, q, v)

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
        self.PrintContacts(t)

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
        self.PrintContacts(t)

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

