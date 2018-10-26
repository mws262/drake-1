import numpy as np
from pydrake.all import (LeafSystem)

class BoxController(LeafSystem):
  def __init__(self):
    self.plan = ManipulationPlan()


  # Loads all plans into the controller.
  def LoadPlans():
    # Read in the plans for the robot.
    plan.ReadRobotQQdotAndQddot(
        "examples/iiwa_soccer/plan/joint_timings_fit.mat",
        "examples/iiwa_soccer/plan/joint_angle_fit.mat",
        "examples/iiwa_soccer/plan/joint_vel_fit.mat",
        "examples/iiwa_soccer/plan/joint_accel_fit.mat")

    # Read in the plans for the point of contact.
    plan.ReadContactPoint("examples/iiwa_soccer/plan/contact_pt_timings.mat",
        "examples/iiwa_soccer/plan/contact_pt_positions.mat",
        "examples/iiwa_soccer/plan/contact_pt_velocities.mat")

    # Read in the plans for the ball kinematics.
    plan.ReadBallQVAndVdot(
        "examples/iiwa_soccer/plan/ball_timings.mat",
        "examples/iiwa_soccer/plan/ball_com_positions.mat",
        "examples/iiwa_soccer/plan/ball_quats.mat",
        "examples/iiwa_soccer/plan/ball_com_velocities.mat",
        "examples/iiwa_soccer/plan/ball_omegas.mat",
        "examples/iiwa_soccer/plan/ball_com_accelerations.mat",
        "examples/iiwa_soccer/plan/ball_alphas.mat",
        "examples/iiwa_soccer/plan/contact_status.mat")


# TODO: create scenegraph+plant context.
# TODO: populate the mapping from geometry IDs to bodies.
# TODO: set geometry_query_input_port

  # Constructs the Jacobian matrices.
  def ConstructJacobians(context, contacts):

    # Get the tree.
    tree = robot_and_ball_plant.tree()

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

    # Get the two body indices.
    for i in range(nc):
      point_pair = contacts[i]

      # Get the surface normal in the world frame.
      n_BA_W = point_pair.nhat_BA_W

      # Get the two bodies.
      geometry_A_id = point_pair.id_A
      geometry_B_id = point_pair.id_B
      body_A_index = geometry_id_to_body_index_.at(geometry_A_id)
      body_B_index = geometry_id_to_body_index_.at(geometry_B_id)
      body_A = tree.get_body(body_A_index)
      body_B = tree.get_body(body_B_index)

      # The reported point on A's surface (As) in the world frame (W).
      p_WAs = point_pair.p_WCa

      # The reported point on B's surface (Bs) in the world frame (W).
      p_WBs = point_pair.p_WCb

      # Get the point of contact in the world frame.
      p_W = (p_WAs + p_WBs) * 0.5

      # Get the geometric Jacobian for the velocity of the contact point
      # as moving with Body A.
      J_WAc = tree.CalcPointsGeometricJacobianExpressedInWorld(
          context, body_A.body_frame(), p_W) 

      # Get the geometric Jacobian for the velocity of the contact point
      # as moving with Body B.
      J_WBc = tree.CalcPointsGeometricJacobianExpressedInWorld(
          context, body_B.body_frame(), p_W) 

      # Compute the linear components of the Jacobian.
      J = J_WAc - J_WBc

      # Compute an orthonormal basis using the contact normal.
      kXAxisIndex = 0, kYAxisIndex = 1, kZAxisIndex = 2
      R_WC = math::ComputeBasisFromAxis(kXAxisIndex, n_BA_W)
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


# TODO: What should the box be doing when it is not supposed to make contact?
# Computes the control torques when contact is not desired.
  def ComputeTorquesForContactNotDesired(context): 
    # Get the desired robot acceleration.
    q_robot_des = plan.GetRobotQQdotAndQddot(
        context.get_time()).head(nv_robot())
    qdot_robot_des = plan.GetRobotQQdotAndQddot(
        context.get_time()).segment(nv_robot(), nv_robot())
    qddot_robot_des = plan.GetRobotQQdotAndQddot(
        context.get_time()).tail(nv_robot())

    # Get the robot current generalized position and velocity.
    q_robot = get_robot_q(context)
    qd_robot = get_robot_qd(context)

    # Set qddot_robot_des using error feedback.
    qddot = qddot_robot_des +
        joint_kp_ * (q_robot_des - q_robot) +
        joint_ki_ * get_integral_value(context) +
        joint_kd_ * (qdot_robot_des - qd_robot)

    # Set the state in the robot context to q_robot and qd_robot. 
    x = robot_mbp.tree().get_mutable_multibody_state_vector(
      robot_context.get())
    assert len(x) == len(q_robot) + len(qd_robot)
    x[0:len(q_robot)-1] = q_robot
    x[-len(qd_robot):] = qd_robot

    # Get the generalized inertia matrix.
    M = robot_mbp.tree().CalcMassMatrixViaInverseDynamics(robot_context)
    lltM = np.linalg.cholesky(M)

    # Compute the contribution from force elements.
    robot_tree = robot_mbp.tree()
    link_wrenches = MultibodyForces(robot_tree)
    pcache = PositionKinematicsCache(robot_tree.get_topology())
    vcache = VelocityKinematicsCache(robot_tree.get_topology())
    robot_tree.CalcPositionKinematicsCache(robot_context, &pcache)
    robot_tree.CalcVelocityKinematicsCache(robot_context, pcache, &vcache)

    # Compute the external forces.
    fext = -robot_tree.CalcInverseDynamics(
        robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

    # Compute inverse dynamics.
    return M * qddot - fext

  # Computes the control torques when contact is desired and the robot and the
  # ball are *not* in contact.
  def ComputeTorquesForContactDesiredButNoContact(context):
    # Get the generalized positions for the robot and the ball.
    q0 = get_all_q(context)
    v0 = get_all_v(context)

    # Get the relevant trees.
    all_tree = robot_and_ball_plant.tree()
    robot_tree = robot_mbp.tree()

    # Set the joint velocities for the robot to zero.
    all_tree.set_velocities_in_array(
        robot_instance, np.zeros([nv_robot, 1]), &v0)

    # Transform the velocities to time derivatives of generalized
    # coordinates.
    qdot0 = BasicVector(robot_and_ball_plant.tree().num_positions())
    robot_and_ball_plant.MapVelocityToQDot(context, v0, &qdot0)

    # TODO(edrumwri): turn this system into an actual discrete system.
    control_freq = 100.0  # 100 Hz.
    dt = 1.0/control_freq

    # Get the estimated position of the ball and the robot at the next time
    # step using a first order approximation to position and the current
    # velocities.
    q1 = q0 + dt * qdot0.CopyToVector()

    # Update the context to use configuration q1 in the query.
    UpdateRobotAndBallConfigurationForGeometricQueries(q1)

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.EvalAbstractInput(
        scenegraph_and_mbp_query_context, geometry_query_input_port).
        GetValue<geometry::QueryObject<double>>()

    # Get the box and the ball bodies.
    ball_body = &get_ball_from_robot_and_ball_tree()
    box_body = &get_box_from_robot_and_ball_tree()
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
      body_A_index = geometry_id_to_body_index_.at(geometry_A_id)
      body_B_index = geometry_id_to_body_index_.at(geometry_B_id)
      body_A = &all_tree.get_body(body_A_index)
      body_B = &all_tree.get_body(body_B_index)
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
    body_A_index = geometry_id_to_body_index_.at(closest.id_A)
    body_B_index = geometry_id_to_body_index_.at(closest.id_B)
    body_A = &all_tree.get_body(body_A_index)
    body_B = &all_tree.get_body(body_B_index)
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
    assert x.size() == q_robot.size() + qd_robot.size()
    x[0:len(q_robot)-1] = q_robot
    x[-len(qd_robot):] = qd_robot

    # Get the geometric Jacobian for the velocity of the closest point on the
    # robot as moving with the robot Body A.
    J_WAc = robot_tree.CalcPointsGeometricJacobianExpressedInWorld(
        robot_context, box_body.body_frame(), closest_Aw)

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
    qddot =
        joint_kp_ * (q_robot_des - q_robot) +
            joint_kd_ * (qdot_robot_des - qd_robot)

    # Get the generalized inertia matrix.
    M = robot_tree.CalcMassMatrixViaInverseDynamics(*robot_context)
    lltM = np.linalg.cholesky(M)

    # Compute the contribution from force elements.
    link_wrenches = MultibodyForces(robot_tree)
    PositionKinematicsCache<double> pcache(robot_tree.get_topology())
    VelocityKinematicsCache<double> vcache(robot_tree.get_topology())
    robot_tree.CalcPositionKinematicsCache(*robot_context, &pcache)
    robot_tree.CalcVelocityKinematicsCache(*robot_context, pcache, &vcache)

    # Compute the external forces.
    fext = -robot_tree.CalcInverseDynamics(
        robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

    # Compute inverse dynamics.
    return M * qddot - fext

  # Constructs the robot actuation matrix.
  def ConstructActuationMatrix(): 


  # Constructs the matrix that zeros angular velocities for the ball (and
  # does not change the linear velocities).
  def ConstructWeightingMatrix(): 


  # Computes the control torques when contact is desired and the robot and the
  # ball are in contact.
  def ComputeTorquesForContactDesiredAndContacting(context, contacts):
    # ***************************************************************
    # Note: This code is specific to this example/system.
    # ***************************************************************

    # Get the number of generalized positions, velocities, and actuators.
    nv = robot_and_ball_plant.tree().num_velocities()
    assert nv == nv_robot() + nv_ball()
    num_actuators = robot_and_ball_plant.tree().num_actuators()
    assert num_actuators == nv_robot()

    # Get the generalized positions and velocities.
    q = get_all_q(context)
    v = get_all_v(context)

    # Construct the actuation and weighting matrices.
#  B = MatrixXd::Zero(nv, num_actuators)
  B = ConstructActuationMatrix()
  P = ConstructWeightingMatrix()

  # Get the generalized inertia matrix.
  M = robot_mbp.tree().CalcMassMatrixViaInverseDynamics(*robot_context)
  lltM = Eigen::LLT(M)
  assert lltM.info() == Eigen::Success

  # Compute the contribution from force elements.
  robot_tree = robot_mbp.tree()
  multibody::MultibodyForces<double> link_wrenches(robot_tree)
  PositionKinematicsCache<double> pcache(robot_tree.get_topology())
  VelocityKinematicsCache<double> vcache(robot_tree.get_topology())
  robot_tree.CalcPositionKinematicsCache(*robot_context, &pcache)
  robot_tree.CalcVelocityKinematicsCache(*robot_context, pcache, &vcache)

  # Compute the external forces.
  fext = -robot_tree.CalcInverseDynamics(
      *robot_context, VectorXd::Zero(nv_robot()), link_wrenches)

  # TODO(edrumwri): Check whether desired ball acceleration is in the right
  # format to match with layout of M.

  # Get the desired ball acceleration.
  vdot_ball_des = plan.GetBallQVAndVdot(context.get_time()).
      tail(nv_ball())

  # Construct the Jacobians and the Jacobians times the velocity.
  jacobians = ConstructJacobians(context, contacts)

  # Get the Jacobians at the point of contact: N, S, T, and construct Z and
  # Zdot_v.
  MatrixXd Z(N.rows() * 3, N.cols())
  Z.topRows(N.rows()) = N
  Z.middleRows(N.rows(), N.rows()) = S
  Z.bottomRows(N.rows()) = T

  # Set the time-derivatives of the Jacobians times the velocity.
  MatrixXd Zdot_v(Ndot_v.rows() * 3, Ndot_v.cols())
  Zdot_v.topRows(Ndot_v.rows()) = Ndot_v
  Zdot_v.middleRows(Ndot_v.rows(), Ndot_v.rows()) = Sdot_v
  Zdot_v.bottomRows(Ndot_v.rows()) = Tdot_v
  assert Zdot_v.cols() == 1

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
  MatrixXd D(nv, num_actuators + nc)
  D.setZero()
  D.topLeftCorner(B.rows(), B.cols()) = B
#  D.bottomRightCorner(Z.cols(), Z.rows()) = Z.T
  D.bottomRightCorner(N.cols(), N.rows()) = N.T

  # Set the Hessian matrix for the QP.
  H = D.T * lltM.solve(P.T) * P *
      lltM.solve(D)
  lltH = Eigen::LDLT(lltH)

  # Verify that the Hessian is positive semi-definite.
  H = H + Eigen::MatrixXd::Identity(H.rows(), H.cols()) * 1e-8
  assert lltH.compute(H).info() == Eigen::Success

  # Compute the linear terms.
  c = D.T * lltM.solve(P.T) * (
      -vdot_ball_des + P * lltM.solve(fext))

  # Set the affine constraint matrix.
#  const MatrixXd A = Z * lltM.solve(D)
#  b = -Z * lltM.solve(fext) - Zdot_v
  MatrixXd A = N * lltM.solve(D)
  b = -N * lltM.solve(fext) - Ndot_v
  assert b.rows() == ndual

  # Prepare to solve the QP using the direct solution to the KKT system.
  MatrixXd K(nprimal + ndual, nprimal + ndual)
  K.block(0, 0, nprimal, nprimal) = H
  K.block(0, nprimal, nprimal, ndual) = -A.T
  K.block(nprimal, ndual, ndual, nprimal).setZero()
  K.block(nprimal, 0, ndual, nprimal) = A

  # Set the right hand side for the KKT solutoin.
  rhs(nprimal + ndual)
  rhs.segment(0, nprimal) = -c
  rhs.segment(nprimal, ndual) = b

  # Solve the KKT system.
  const Eigen::PartialPivLU<MatrixXd> lu(K)
  z = lu.solve(rhs)

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
cf = z.tail(nc)
  # Output some logging information.
  vdot = lltM.solve(D*z.head(nprimal) + fext)
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
  print "torque: " + z.head(nv_robot()).T

  # First nv_robot() primal variables are the torques.
  return z.head(nv_robot())


  # Gets the vector of contacts.
  def FindContacts():
    # TODO: Update the state of the query context to that in the true context.

    # Get the tree corresponding to all bodies.
    all_tree = robot_and_ball_plant.tree()

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.EvalAbstractInput(
        *scenegraph_and_mbp_query_context, geometry_query_input_port)->
        GetValue<geometry::QueryObject<double>>()

    # Determine the set of contacts.
    contacts = query_object.ComputePointPairPenetration()

    # Get the ball body and foot bodies.
    ball_body = &get_ball_from_robot_and_ball_tree()
    box_body = &get_box_from_robot_and_ball_tree()
    world_body = &get_world_from_robot_and_ball_tree()

    # Make sorted pairs to check.
    ball_box_pair = MakeSortedPair(ball_body, box_body)
    ball_world_pair = MakeSortedPair(ball_body, world_body)

    # Remove contacts between all but the robot foot and the ball and the
    # ball and the ground.
#    for i in range(len(contacts)):
    for (int i = 0 i < static_cast<int>(contacts.size()) ++i) {
    geometry_A_id = contacts[i].id_A
    geometry_B_id = contacts[i].id_B
    ody_A_index = geometry_id_to_body_index_.at(geometry_A_id)
    ody_B_index = geometry_id_to_body_index_.at(geometry_B_id)
    body_a = &all_tree.get_body(body_A_index)
    body_b = &all_tree.get_body(body_B_index)
    body_a_b_pair = MakeSortedPair(body_a, body_b)
    if (body_a_b_pair != ball_box_pair && body_a_b_pair != ball_world_pair) {
      contacts[i] = contacts.back()
      contacts.pop_back()
      --i
    } else {
#      std::cout << "Contact detected between " << body_a->get_name() <<
#                " and " << body_b->get_name() << std::endl
    }
  }

  return contacts

# Calculate what torques to apply to the joints.
  def DoControlCalc(context):
    # Determine whether we're in a contacting or not-contacting phase.
    contact_desired = plan.IsContactDesired(context.get_time())

    # Get the number of generalized positions, velocities, and actuators.
    nv = robot_and_ball_plant.tree().num_velocities()
    assert nv == nv_robot() + nv_ball()
    num_actuators = robot_and_ball_plant.tree().num_actuators()
    assert num_actuators == nv_robot()

    # Get the generalized positions and velocities.
    q = get_all_q(context)
    v = get_all_v(context)

    # Compute tau.
    if contact_desired == True:
      # Find contacts.
      contacts = FindContacts()

      # Get the number of points of contact.
      nc = len(contacts)

      # Two cases: in the first, the robot and the ball are already in contact,
      # as desired. In the second, the robot desires to be in contact, but the
      # ball and robot are not contacting: the robot must intercept the ball.
      if nc >= 2:
        tau = ComputeTorquesForContactDesiredAndContacting(context, contacts)
      else:
        tau = ComputeTorquesForContactDesiredButNoContact(context)
    else:
      # No contact desired.
      tau = ComputeTorquesForContactNotDesired(context)

    # Set the torque output.
    torque_out = BasicVector(tau)
    output->SetFrom(torque_out)

  # Gets the value of the integral term in the state.
  def get_integral_value(const Context<double>& context): 
    return context.get_continuous_state_vector().CopyToVector()

  # Sets the value of the integral term in the state.
  def set_integral_value(
      Context<double>* context, qint): 
    assert qint.size() == nv_robot()
    context->get_mutable_continuous_state_vector().SetFromVector(qint)

  def DoCalcTimeDerivatives(context, derivatives): 
    # Determine whether we're in a contacting or not-contacting phase.
    contact_intended = plan.IsContactDesired(context.get_time())

    if contact_intended:
      derivatives->get_mutable_vector().SetFromVector(VectorXd::Zero(nv_robot()))
    else:
      # Get the desired robot configuration.
      q_robot_des = plan.GetRobotQQdotAndQddot(
          context.get_time()).head(nv_robot())

      # Get the current robot configuration.
      x = dynamic_cast<const BasicVector<double>&>(
          context.get_continuous_state_vector()).get_value()
      q_robot = robot_and_ball_plant.tree().
          get_positions_from_array(robot_instance, x)
      derivatives->get_mutable_vector().SetFromVector(q_robot_des - q_robot)

  # Gets the ball body from the robot and ball tree.
  def get_ball_from_robot_and_ball_tree():
    return robot_and_ball_plant.tree().GetBodyByName("ball") 

  # Gets the box link from the robot and ball tree.
  def get_box_from_robot_and_ball_tree(): 
    return robot_and_ball_plant.tree().GetBodyByName("box") 

  # Gets the world body from the robot and ball tree.
  def get_world_from_robot_and_ball_tree(): 
    return robot_and_ball_plant.tree().world_body()

  def get_all_q(const Context<double>& context):
    robot_q = get_robot_q(context)
    ball_q = get_ball_q(context)
    VectorXd q(nq_ball() + nv_robot())

    # Sanity check.
    for i in range(len(q)):
      q[i] = float("nan")

    all_tree = robot_and_ball_plant.tree()
    all_tree.set_positions_in_array(robot_instance, robot_q, &q)
    all_tree.set_positions_in_array(ball_instance_, ball_q, &q)

    # Sanity check.
    for i in len(q):
      assert !math::isnan(q[i])

    return q


  def get_all_v(const Context<double>& context):
    robot_qd = get_robot_qd(context)
    ball_v = get_ball_v(context)
    VectorXd v(nv_ball() + nv_robot())

    # Sanity check.
    for i in range(len(v)): 
      v[i] = float("nan") 

    all_tree = robot_and_ball_plant.tree()
    all_tree.set_velocities_in_array(robot_instance, robot_qd, &v)
    all_tree.set_velocities_in_array(ball_instance_, ball_v, &v)
    return v


  def DoPublish(context,
    const std::vector<const PublishEvent<double>*>&): 



