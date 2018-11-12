# TODO: address remaining TODOs
import numpy as np
from manipulation_plan import ManipulationPlan

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType, BasicVector)

class BoxController(LeafSystem):
  def __init__(self, all_plant, robot_plant, mbw, kp, kd, robot_gv_kp, robot_gv_ki, robot_gv_kd, robot_instance, ball_instance):
    LeafSystem.__init__(self)

    # Construct the plan.
    self.plan = ManipulationPlan()
    self.LoadPlans()

    # Save the robot and ball instances.
    self.robot_instance = robot_instance
    self.ball_instance = ball_instance

    # Get the plants.
    self.robot_plant = robot_plant
    self.robot_and_ball_plant = all_plant
    self.mbw = mbw

    # Get the number of actuators.
    self.command_output_size = self.robot_plant.num_actuators()

    # Create contexts.
    self.mbw_context = mbw.CreateDefaultContext()
    self.robot_context = robot_plant.CreateDefaultContext()

    # Set PID gains.
    self.cartesian_kp = kp
    self.cartesian_kd = kd
    self.robot_gv_kp = robot_gv_kp
    self.robot_gv_ki = robot_gv_ki
    self.robot_gv_kd = robot_gv_kd

    # Declare states and ports.
    self._DeclareContinuousState(self.nq_robot())   # For integral control state.
    self.input_port_index_estimated_robot_q = self._DeclareInputPort(
        PortDataType.kVectorValued, self.nq_robot()).get_index()
    self.input_port_index_estimated_robot_qd = self._DeclareInputPort(
        PortDataType.kVectorValued, self.nv_robot()).get_index()
    self.input_port_index_estimated_ball_q = self._DeclareInputPort(
        PortDataType.kVectorValued, self.nq_ball()).get_index()
    self.input_port_index_estimated_ball_v = self._DeclareInputPort(
        PortDataType.kVectorValued, self.nv_ball()).get_index()
    self._DeclareVectorOutputPort(
        BasicVector(self.command_output_size),
        self.DoControlCalc) # Output 0.

  def get_input_port_estimated_robot_q(self):
    return self.get_input_port(self.input_port_index_estimated_robot_q)

  def get_input_port_estimated_robot_qd(self):
    return self.get_input_port(self.input_port_index_estimated_robot_qd)

  def get_input_port_estimated_ball_q(self):
    return self.get_input_port(self.input_port_index_estimated_ball_q)

  def get_input_port_estimated_ball_v(self):
    return self.get_input_port(self.input_port_index_estimated_ball_v)

  # Gets the number of robot degrees of freedom.
  def nq_robot(self):
    return self.robot_plant.num_positions()

  # Gets the number of robot velocity variables.
  def nv_robot(self):
    return self.robot_plant.num_velocities()

  # Gets the number of ball degrees of freedom.
  def nq_ball(self):
    return self.robot_and_ball_plant.num_positions() - self.nq_robot()

  # Gets the number of ball velocity variables.
  def nv_ball(self):
    return self.robot_and_ball_plant.num_velocities() - self.nv_robot()

  # Gets the robot configuration.
  def get_robot_q(self, context):
    return self.EvalVectorInput(self.input_port_index_estimated_robot_q)

  # Gets the ball configuration.
  def get_ball_q(self, context):
    return self.EvalVectorInput(self.input_port_index_estimated_ball_q)

  # Gets the robot velocity.
  def get_robot_qd(self, context):
    return self.EvalVectorInput(self.input_port_index_estimated_robot_qd)

  # Gets the ball velocity
  def get_ball_qd(self, context):
    return self.EvalVectorInput(self.input_port_index_estimated_ball_v)

  # Makes a sorted pair.
  def MakeSortedPair(self, a, b):
    if b > a:
      return (b, a)
    else:
      return (a, b)

  # Loads all plans into the controller.
  def LoadPlans(self):
    # Read in the plans for the robot.
    self.plan.ReadRobotQQdotAndQddot("plan/joint_timings_fit.mat",
                                     "plan/joint_angle_fit.mat",
                                     "plan/joint_vel_fit.mat",
                                     "plan/joint_accel_fit.mat")

    # Read in the plans for the point of contact.
    self.plan.ReadContactPoint("plan/contact_pt_timings.mat",
        "plan/contact_pt_positions.mat",
        "plan/contact_pt_velocities.mat")

    # Read in the plans for the ball kinematics.
    self.plan.ReadBallQVAndVdot(
        "plan/ball_timings.mat",
        "plan/ball_com_positions.mat",
        "plan/ball_quats.mat",
        "plan/ball_com_velocities.mat",
        "plan/ball_omegas.mat",
        "plan/ball_com_accelerations.mat",
        "plan/ball_alphas.mat",
        "plan/contact_status.mat")


  # Constructs the Jacobian matrices.
  def ConstructJacobians(self, mbw_context, inspector, contacts):

    # Get the robot and ball multibody-plant context.
    robot_and_ball_context = self.mbw.GetMutableSubsystemContext(
      self.robot_and_ball_plant, mbw_context)

    # Get the tree.
    tree = self.robot_and_ball_plant.tree()

    # Get the numbers of contacts and generalized velocities.
    nc = len(contacts) 

    # Set the number of generalized velocities.
    nv = tree.num_velocities() 

    # Size the matrices.
    N = np.empty([nc, nv])
    S = np.empty([nc, nv])
    T = np.empty([nc, nv])
    Ndot_v = np.empty([nc, 1])
    Sdot_v = np.empty([nc, 1])
    Tdot_v = np.empty([nc, 1])

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.EvalAbstractInput(
      mbw_context, self.geometry_query_input_port).GetValue()
    #GetValue<geometry::QueryObject<double>>()
    inspector = query_object.inspector()

    # Get the two body indices.
    for i in range(nc):
      point_pair = contacts[i]

      # Get the surface normal in the world frame.
      n_BA_W = point_pair.nhat_BA_W

      # Get the two bodies.
      geometry_A_id = point_pair.id_A
      geometry_B_id = point_pair.id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_tree.GetBodyFromFrameId(frame_A_id)
      body_B = all_tree.GetBodyFromFrameId(frame_B_id)

      # The reported point on A's surface (A) in the world frame (W).
      pr_WA = point_pair.p_WCa

      # The reported point on B's surface (B) in the world frame (W).
      pr_WB = point_pair.p_WCb

      # Get the point of contact in the world frame.
      pc_W = (p_WA + p_WB) * 0.5

      # Transform pr_W to the body frames.
      X_wa = all_tree.EvalBodyPoseInWorld(
          scenegraph_and_mbp_query_context, body_A)
      X_wb = all_tree.EvalBodyPoseInWorld(
          scenegraph_and_mbp_query_context, body_B)
      p_A = X_wa.inverse() * pc_W
      p_B = X_wb.inverse() * pc_W

      # Get the geometric Jacobian for the velocity of the contact point
      # as moving with Body A.
      J_WAc = tree.CalcPointsGeometricJacobianExpressedInWorld(
          robot_and_ball_context, body_A.body_frame(), p_A)

      # Get the geometric Jacobian for the velocity of the contact point
      # as moving with Body B.
      J_WBc = tree.CalcPointsGeometricJacobianExpressedInWorld(
          robot_and_ball_context, body_B.body_frame(), p_B)

      # Compute the linear components of the Jacobian.
      J = J_WAc - J_WBc

      # Compute an orthonormal basis using the contact normal.
      kXAxisIndex = 0
      kYAxisIndex = 1
      kZAxisIndex = 2
      R_WC = ComputeBasisFromAxis(kXAxisIndex, n_BA_W)
      t1_BA_W = R_WC[:,kYAxisIndex]
      t2_BA_W = R_WC[:,kZAxisIndex]

      # Set N, S, and T.
      N[i,:] = n_BA_W.T * J
      S[i,:] = t1_BA_W.T * J
      T[i,:] = t2_BA_W.T * J

      # TODO: Set Ndot_v, Sdot_v, Tdot_v properly.
      Ndot_v *= 0 # = n_BA_W.T * Jdot_v
      Sdot_v *= 0 # = t1_BA_W.T * Jdot_v
      Tdot_v *= 0 # = t2_BA_W.T * Jdot_v

    return N, S, T, Ndot_v, Sdot_v, Tdot_v


# TODO: What should the box be doing when it is not supposed to make contact?
# Computes the control torques when contact is not desired.
  def ComputeActuationForContactNotDesired(self, context):
    # Get the desired robot acceleration.
    q_robot_des = self.plan.GetRobotQQdotAndQddot(
        context.get_time())[0:nv_robot()-1]
    qdot_robot_des = self.plan.GetRobotQQdotAndQddot(
        context.get_time())[nv_robot(), 2*nv_robot()-1]
    qddot_robot_des = self.plan.GetRobotQQdotAndQddot(
        context.get_time())[-nv_robot():]

    # Get the robot current generalized position and velocity.
    q_robot = get_robot_q(context)
    qd_robot = get_robot_qd(context)

    # Set qddot_robot_des using error feedback.
    qddot = qddot_robot_des + self.robot_gv_kp * (q_robot_des - q_robot) + self.robot_gv_ki * get_integral_value(context) + self.robot_gv_kd * (qdot_robot_des - qd_robot)

    # Set the state in the robot context to q_robot and qd_robot. 
    x = self.robot_plant.tree().get_mutable_multibody_state_vector(
      robot_context)
    assert len(x) == len(q_robot) + len(qd_robot)
    x[0:len(q_robot)-1] = q_robot
    x[-len(qd_robot):] = qd_robot

    # Get the generalized inertia matrix.
    M = self.robot_plant.tree().CalcMassMatrixViaInverseDynamics(robot_context)
    lltM = np.linalg.cholesky(M)

    # Compute the contribution from force elements.
    robot_tree = self.robot_plant.tree()
    link_wrenches = MultibodyForces(robot_tree)

    # Compute the external forces.
    fext = -robot_tree.CalcInverseDynamics(
        robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

    # Compute inverse dynamics.
    return M * qddot - fext


  def UpdateRobotAndBallConfigurationForGeometricQueries(self, q):

    # Get our own mutable context for the plant.
    robot_and_ball_context = self.mbw.GetMutableSubsystemContext(
      self.robot_and_ball_plant, self.mbw_context)

    # Get the state.
    qvec = self.robot_and_ball_plant.tree().get_mutable_multibody_positions_vector(robot_and_ball_context)
    qvec  = q


  # Computes the control torques when contact is desired and the robot and the
  # ball are *not* in contact.
  def ComputeActuationForContactDesiredButNoContact(self, context):
    # Get the generalized positions for the robot and the ball.
    q0 = get_all_q(context)
    v0 = get_all_v(context)

    # Get the relevant trees.
    all_tree = self.robot_and_ball_plant.tree()
    robot_tree = self.robot_plant.tree()

    # Set the joint velocities for the robot to zero.
    all_tree.set_velocities_in_array(
        self.robot_instance, np.zeros([nv_robot, 1]), v0)

    # Transform the velocities to time derivatives of generalized
    # coordinates.
    qdot0 = self.robot_and_ball_plant.MapVelocityToQDot(context, v0)

    # TODO(edrumwri): turn this system into an actual discrete system.
    control_freq = 100.0  # 100 Hz.
    dt = 1.0/control_freq

    # Get the estimated position of the ball and the robot at the next time
    # step using a first order approximation to position and the current
    # velocities.
    q1 = q0 + dt * qdot0.CopyToVector()

    # Update the context to use configuration q1 in the query. This will modify
    # the mbw context, used immediately below.
    UpdateRobotAndBallConfigurationForGeometricQueries(q1)

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.EvalAbstractInput(
        self.mbw_context, geometry_query_input_port).GetValue()
        #        GetValue<geometry::QueryObject<double>>()
    inspector = query_object.inspector()

    # Get the box and the ball bodies.
    ball_body = get_ball_from_robot_and_ball_tree()
    box_body = get_box_from_robot_and_ball_tree()
    box_and_ball = MakeSortedPair(ball_body, box_body)

    # Get the closest points on the robot foot and the ball corresponding to q1
    # and v0.
    closest_points = query_object.ComputeSignedDistancePairwiseClosestPoints()
    assert len(closest_points) > 0
    found_index = -1
    for i in range(len(closest_points)):
      # Get the two bodies in contact.
      point_pair = closest_points[i]
      geometry_A_id = point_pair.id_A
      geometry_B_id = point_pair.id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_tree.GetBodyFromFrameId(frame_A_id)
      body_B = all_tree.GetBodyFromFrameId(frame_B_id)
      bodies = MakeSortedPair(body_A, body_B)

      # If the two bodies correspond to the foot (box) and the ball, mark the
      # found index and stop looping.
      if bodies == box_and_ball:
        found_index = i
        break

    # Get the signed distance data structure.
    assert found_index >= 0
    closest = closest_points[found_index]

    # Make A be the body belonging to the robot.
    geometry_A_id = closest.id_A
    geometry_B_id = closest.id_B
    frame_A_id = inspector.GetFrameId(geometry_A_id)
    frame_B_id = inspector.GetFrameId(geometry_B_id)
    body_A = all_tree.GetBodyFromFrameId(frame_A_id)
    body_B = all_tree.GetBodyFromFrameId(frame_B_id)
    if body_A != ball_body:
      # Swap A and B.
      body_A, body_B = body_B, body_A
      closest.id_A, closest.id_B = closest.id_B, closest.id_A
      closest.p_ACa, closest.p_BCb = closest.p_BCb, closest.p_ACa

    # Get the closest points on the bodies. They'll be in their respective body
    # frames.
    closest_Aa = closest.p_ACa
    closest_Bb = closest.p_BCb

    # Transform the points in the body frames corresponding to q1 to the
    # world frame.
    X_wa = all_tree.EvalBodyPoseInWorld(
        scenegraph_and_mbp_query_context, body_A)
    X_wb = all_tree.EvalBodyPoseInWorld(
        scenegraph_and_mbp_query_context, body_B)
    closest_Aw = X_wa * closest_Aa
    closest_Bw = X_wb * closest_Bb

    # Get the vector from the closest point on the foot to the closest point
    # on the ball in the body frames.
    linear_v_des = (closest_Bw - closest_Aw) / dt

    # Get the robot current generalized position and velocity.
    q_robot = get_robot_q(context)
    qd_robot = get_robot_qd(context)

    # Set the state in the robot context to q_robot and qd_robot.
    x = robot_tree.get_mutable_multibody_state_vector(robot_context)
    assert x.shape[0] == q_robot.size() + qd_robot.size()
    x[0:len(q_robot)-1] = q_robot
    x[-len(qd_robot):] = qd_robot

    # Get the geometric Jacobian for the velocity of the closest point on the
    # robot as moving with the robot Body A.
    J_WAc = robot_tree.CalcPointsGeometricJacobianExpressedInWorld(
        robot_context, box_body.body_frame(), closest_Aa)

    # Set the rigid body constraints.
    '''
    num_constraints = 1
    WorldPositionConstraint constraint()

    # Set the inverse kinematics options.
    IKoptions ik_options

    # Compute the desired robot configuration.
    q_robot_des = q_robot
    RigidBodyTree<double>* robot_tree_nc =
        const_cast<RigidBodyTree<double>*>(&robot_tree_)
    int info
    std::vector<std::string> infeasible_constraint
    inverseKin(robot_tree_nc, q_robot_des, q_robot_des,
               num_constraints, constraint_array, ik_options,
               &q_robot_des, &info, &infeasible_constraint)

    # Just use the current configuration if not successful.
    if (info > 6)
      q_robot_des = q_robot
    '''
    q_robot_des = q_robot

    # Use resolved-motion rate control to determine the robot velocity that
    # would be necessary to realize the desired end-effector velocity.
    qdot_robot_des = np.linalg.lstsq(J_WAc, linear_v_des)

    # Set qddot_robot_des using purely error feedback.
    qddot = self.robot_gv_kp * (q_robot_des - q_robot) + self.robot_gv_kd * (qdot_robot_des - qd_robot)

    # Get the generalized inertia matrix.
    M = robot_tree.CalcMassMatrixViaInverseDynamics(self.robot_context)
    lltM = np.linalg.cholesky(M)

    # Compute the contribution from force elements.
    link_wrenches = MultibodyForces(robot_tree)

    # Compute the external forces.
    fext = -robot_tree.CalcInverseDynamics(
        robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

    # Compute inverse dynamics.
    return M * qddot - fext

  # Constructs the robot actuation matrix.
  def ConstructRobotActuationMatrix(self):
    # We assume that each degree of freedom is actuatable. There is no way to
    # verify this because we want to be able to use free bodies as "robots" too.

    # Get the robot and ball multibody-plant context.
    robot_and_ball_context = self.mbw.GetMutableSubsystemContext(
      self.robot_and_ball_plant, self.mbw_context)

    # First zero out the generalized velocities for the whole multibody.
    x = self.robot_plant.tree().get_mutable_multibody_state_vector(
      robot_context)
    x[-len(all_v):] = zeros([nv_robot() + nv_ball(), 1])

    # Now set the velocities in the generalized velocity array to non-zero
    # values.
    self.robot_plant.tree().set_velocities_in_array(
      self.robot_instance,
      np.ones([nv_robot(), 1]),
      x[-len(all_v):])

    # The matrix is of size nv_robot() + nv_ball() x nv_ball().
    B = np.zeros([nv_robot() + nv_ball(), nv_ball()])

    return B


  # Constructs the matrix that zeros angular velocities for the ball (and
  # does not change the linear velocities).
  def ConstructBallStateWeightingMatrix(self):
    # The ball's quaternionxyzmobilizer permits movement along all six DOF.
    # We need the location in the array corresponding to angular motions.

    '''
    # Get the number of velocities.
    nv = nv_ball() + nv_robot

    # The size of the returned matrix will be:
    W = np.zeros([nv, nv])

    # Get the robot and ball multibody-plant context.
    robot_and_ball_context = self.mbw.GetMutableSubsystemContext(
      self.robot_and_ball_plant, self.mbw_context)

    # Zero out the generalized velocities for the whole multibody.
    x = self.robot_and_ball_plant.tree().get_mutable_multibody_state_vector(
      robot_context)
    x[-nv:] = zeros([nv, 1])

    # Find the quaternion mobilizer.
    for i in range(len(self.robot_and_ball_plant.tree().num_mobilizers())):
      # Get the i'th mobilizer.
      mobilizer = self.robot_and_ball_plant.tree().get_mobilizer(i)

      # Get the outer frame body.
      outboard_body = mobilizer.outboard_body()

      # See whether the body matches the one that we are looking for.
      if outboard_body == self.ball_name:
        # A match! Set qmobilizer and break.
        qmobilizer = QuaternionFloatingMobilizer(mobilizer)
        break

    # Set the angular velocity for the ball mobilizer to [1, 2, 3].
    qmobilizer.set_angular_velocity(robot_and_ball_context, np.ones([3, 1]))

    # Identify the non-zero coordinates in the array, and set those elements
    # to 1.0 on the diagonal of the matrix. Set all other entries in the matrix
    # to zero.
    num_non_zero = 0
    W = zeros([nv_ball(), nv_ball()])
    q_len = len(x) - nv
    for i in range(q_len, len(x)):
      if math.abs(x[i]) > 0:
        num_non_zero += 1
        v_index = i - q_len
        W[v_index, v_index] = 1.0

    # Ensure that there were exactly three nonzero entries.
    assert num_non_zero == 3
    '''

    # TODO: Replace this hack with the code above when the mobilizers are bound.
    angular_velocity_starting_index_in_v = 9
    index = angular_velocity_starting_index_in_v
    nv = nv_ball() + nv_robot
    W = np.zeros([nv, nv])
    W[index + 0, index + 0] = 1.0
    W[index + 1, index + 1] = 1.0
    W[index + 2, index + 2] = 1.0

    return W


  # Computes the control torques when contact is desired and the robot and the
  # ball are in contact.
  def ComputeActuationForContactDesiredAndContacting(self, context, contacts):
    # ***************************************************************
    # Note: This code is specific to this example/system.
    # ***************************************************************

    # Get the number of generalized positions, velocities, and actuators.
    nv = self.robot_and_ball_plant.tree().num_velocities()
    assert nv == nv_robot() + nv_ball()
    num_actuators = self.robot_and_ball_plant.tree().num_actuators()
    assert num_actuators == nv_robot()

    # Get the generalized positions and velocities.
    q = get_all_q(context)
    v = get_all_v(context)

    # Construct the actuation and weighting matrices.
    #  B = MatrixXd::Zero(nv, num_actuators)
    B = ConstructRobotActuationMatrix()
    P = ConstructBallStateWeightingMatrix()

    # Get the robot and ball multibody-plant context.
    robot_and_ball_context = self.mbw.GetMutableSubsystemContext(
      self.robot_and_ball_plant, self.mbw_context)

    # Set the state in the robot context.
    x = self.robot_plant.tree().get_mutable_multibody_state_vector(
      robot_and_ball_context)
    x[0:len(v)-1] = q
    x[-len(v):] = v

    # Get the generalized inertia matrix and compute its Cholesky factorization.
    M = self.robot_and_ball_plant.tree().CalcMassMatrixViaInverseDynamics(
      robot_and_ball_context)
    lltM = np.linalg.cholesky(M)

    # Compute the contribution from force elements.
    robot_tree = self.robot_plant.tree()
    link_wrenches = MultibodyForces(robot_tree)

    # Compute the external forces.
    fext = -robot_tree.CalcInverseDynamics(robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

    # TODO(edrumwri): Check whether desired ball acceleration is in the right
    # format to match with layout of M.

    # Get the desired ball acceleration.
    vdot_ball_des = self.plan.GetBallQVAndVdot(context.get_time())[-nv_ball():]

    # Construct the Jacobians and the Jacobians times the velocity.
    N, S, T, Ndot_v, Sdot_v, Tdot_v = ConstructJacobians(context, contacts)

    # Get the Jacobians at the point of contact: N, S, T, and construct Z and
    # Zdot_v.
    Z = np.zeros([N.shape[0] * 3, N.shape[1]])
    Z[0,N.shape[0]-1,:] = N
    Z[N.shape[0]:N.shape[0]*2-1,:] = S
    Z[-N.shape[0]:,:] = T

    # Set the time-derivatives of the Jacobians times the velocity.
    Zdot_v = np.zeros([Ndot_v.shape[0] * 3, Ndot_v.shape[1]])
    Zdot_v[0:Ndot_v.shape[0],:] = Ndot_v
    Zdot_v[Ndot_v.shape[0]:Ndot_v.shape[0]*2-1,:] = Sdot_v
    Zdot_v[-Ndot_v.shape[0],:] = Tdot_v
    assert Zdot_v.shape[1] == 1

    # Primal variables are motor torques and contact force magnitudes.
    nc = len(contacts)
    #  nprimal = num_actuators + nc * 3
    nprimal = num_actuators + nc

    # Dual variables (Lagrange multipliers) correspond to number of linear
    # constraint equations.
    #  ndual = nc * 3
    ndual = nc

    # Construct the matrices necessary to construct the Hessian.
    #  MatrixXd D(nv, num_actuators + nc * 3)
    D = np.zeros([nv, num_actuators + nc])
    D[0:B.shape[0]-1, 0:B.shape[1]-1] = B
    #  D.bottomRightCorner(Z.shape[1], Z.shape[0]) = Z.T
    D[-N.shape[1]:, -N.shape[0]:] = N.T

    # Set the Hessian matrix for the QP.
    H = D.T * np.linalg.cho_solve(lltM, P.T) * P * np.linalg.cho_solve(lltM, D)

    # Verify that the Hessian is positive semi-definite.
    H = H + np.eye(H.shape[0]) * 1e-8
    np.linalg.cholesky(H)

    # Compute the linear terms.
    c = D.T * np.linalg.cho_solve(lltM, P.T) * (-vdot_ball_des + P * np.linalg.cho_solve(lltM, fext))

    # Set the affine constraint matrix.
    #  const MatrixXd A = Z * np.linalg.cho_solve(lltM, D)
    #  b = -Z * np.linalg.cho_solve(lltM, fext) - Zdot_v
    A = N * np.linalg.cho_solve(lltM, D)
    b = -N * np.linalg.cho_solve(lltM, fext) - Ndot_v
    assert b.shape[0] == ndual

    # Prepare to solve the QP using the direct solution to the KKT system.
    K = np.zeros([nprimal + ndual, nprimal + ndual])
    K[0:nprimal-1,0:nprimal-1] = H
    K[0:nprimal-1,nprimal:nprimal+ndual-1] = -A.T
    K[nprimal:nprimal+ndual-1,0:nprimal-1] = A

    # Set the right hand side for the KKT solutoin.
    rhs = np.zeros([nprimal + ndual, 1])
    rhs[0:nprimal-1] = -c
    rhs[nprimal:nprimal+ndual-1] = b

    # Solve the KKT system.
    z = np.linalg.solve(K, rhs)

    # Verify that the solution is reasonably accurate.
    tol = 1e-8
    soln_err = (K * z - rhs).norm()
    assert soln_err < tol

    '''
    # Check whether the computed normal contact forces are tensile.
    cf = z.tail(nc * 3)
    bool tensile = false
    for (int i = 0 i < cf.size() i += 3) {
      if (cf[i] < 0.0) {
        tensile = true
        break
      }
    }
    '''

    # TODO: Temporary- fix me!
    cf = z[-nc:]
    # Output some logging information.
    vdot = np.linalg.cho_solve(lltM, D*z[0:nprimal-1] + fext)
    P_vdot = P * vdot
    print "N * v: " + (N * v).T
    print "S * v: " + (S * v).T
    print "T * v: " + (T * v).T
    print "Ndot * v: " + Ndot_v
    print "Zdot * v: " + Zdot_v
    print "fext: " + fext.T
    print "M: " + M
    print "P: " + P
    print "D: " + D
    print "B: " + B
    print "N: " + N
    print "Z: " + Z
    print "contact forces: " + cf.T
    print "vdot: " + vdot.T
    print "vdot (desired): " + vdot_ball_des.T
    print "P * vdot: " + P_vdot.T
    print "torque: " + z[0:nv_robot-1].T

    # First nv_robot() primal variables are the torques.
    return z[0:nv_robot()-1]


  # Gets the vector of contacts.
  def FindContacts(self, all_q):
    # Set q in the context.
    UpdateRobotAndBallConfigurationForGeometricQueries(all_q)

    # Get the tree corresponding to all bodies.
    all_tree = self.robot_and_ball_plant.tree()

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.EvalAbstractInput(
        mbw_context, geometry_query_input_port).GetValue()
        #GetValue<geometry::QueryObject<double>>()
    inspector = query_object.inspector()

    # Determine the set of contacts.
    contacts = query_object.ComputePointPairPenetration()

    # Get the ball body and foot bodies.
    ball_body = get_ball_from_robot_and_ball_tree()
    box_body = get_box_from_robot_and_ball_tree()
    world_body = get_world_from_robot_and_ball_tree()

    # Make sorted pairs to check.
    ball_box_pair = MakeSortedPair(ball_body, box_body)
    ball_world_pair = MakeSortedPair(ball_body, world_body)

    # Remove contacts between all but the robot foot and the ball and the
    # ball and the ground.
    i = 0
    while i < len(contacts):
      geometry_A_id = contacts[i].id_A
      geometry_B_id = contacts[i].id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_tree.GetBodyFromFrameId(frame_A_id)
      body_B = all_tree.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = MakeSortedPair(body_A, body_B)
      if body_A_B_pair != ball_box_pair and body_A_B_pair != ball_world_pair:
        contacts[i] = contacts[-1]
        del contacts[-1]
      else:
        i += 1
        #      std::cout << "Contact detected between " << body_a->get_name() <<
        #                " and " << body_b->get_name() << std::endl

    return contacts


  # Calculate what torques to apply to the joints.
  def DoControlCalc(self, context):
    # Determine whether we're in a contacting or not-contacting phase.
    contact_desired = self.plan.IsContactDesired(context.get_time())

    # Get the generalized positions.
    q = get_all_q(context)

    # Compute tau.
    if contact_desired == True:
      # Find contacts.
      contacts = FindContacts(q)

      # Get the number of points of contact.
      nc = len(contacts)

      # Two cases: in the first, the robot and the ball are already in contact,
      # as desired. In the second, the robot desires to be in contact, but the
      # ball and robot are not contacting: the robot must intercept the ball.
      if nc >= 2:
        tau = ComputeActuationForContactDesiredAndContacting(context, contacts)
      else:
        tau = ComputeActuationForContactDesiredButNoContact(context)
    else:
      # No contact desired.
      tau = ComputeActuationForContactNotDesired(context)

    # Set the torque output.
    torque_out = BasicVector(tau)
    output.SetFrom(torque_out)


  # Gets the value of the integral term in the state.
  def get_integral_value(self, context):
    return context.get_continuous_state_vector().CopyToVector()


  # Sets the value of the integral term in the state.
  def set_integral_value(self, context, qint):
    assert len(qint) == nv_robot()
    context.get_mutable_continuous_state_vector().SetFromVector(qint)


  def DoCalcTimeDerivatives(self, context, derivatives):
    # Determine whether we're in a contacting or not-contacting phase.
    contact_intended = self.plan.IsContactDesired(context.get_time())

    if contact_intended:
      derivatives.get_mutable_vector().SetFromVector(np.zeros([nv_robot(), 1]))
    else:
      # Get the desired robot configuration.
      q_robot_des = self.plan.GetRobotQQdotAndQddot(
          context.get_time())[0:nv_robot()-1]

      # Get the current robot configuration.
      q_robot = get_robot_q(context)
      derivatives.get_mutable_vector().SetFromVector(q_robot_des - q_robot)


  # Gets the ball body from the robot and ball tree.
  def get_ball_from_robot_and_ball_tree(self):
    return self.robot_and_ball_plant.tree().GetBodyByName("ball")


  # Gets the box link from the robot and ball tree.
  def get_box_from_robot_and_ball_tree(self):
    return self.robot_and_ball_plant.tree().GetBodyByName("box")


  # Gets the world body from the robot and ball tree.
  def get_world_from_robot_and_ball_tree(self):
    return self.robot_and_ball_plant.tree().world_body()


  def get_all_q(self, context):
    robot_q = get_robot_q(context)
    ball_q = get_ball_q(context)
    q = np.zeros([nq_ball() + nv_robot(), 1])

    # Sanity check.
    for i in range(len(q)):
      q[i] = float("nan")

    all_tree = self.robot_and_ball_plant.tree()
    all_tree.set_positions_in_array(robot_instance, robot_q, q)
    all_tree.set_positions_in_array(ball_instance_, ball_q, q)

    # Sanity check.
    for i in len(q):
      assert not math.isnan(q[i])

    return q


  def get_all_v(self, context):
    robot_qd = get_robot_qd(context)
    ball_v = get_ball_v(context)
    v = np.zeros([nv_ball() + nv_robot(), 1])

    # Sanity check.
    for i in range(len(v)): 
      v[i] = float("nan") 

    all_tree = self.robot_and_ball_plant.tree()
    all_tree.set_velocities_in_array(robot_instance, robot_qd, v)
    all_tree.set_velocities_in_array(ball_instance_, ball_v, v)
    return v


#  def DoPublish(context, publish_events):
