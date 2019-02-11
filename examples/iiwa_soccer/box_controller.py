# TODO: turn this system into an actual discrete system (estimated time required: 30m)
import numpy as np
from manipulation_plan import ManipulationPlan
from embedded_box_soccer_sim import EmbeddedSim

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType,
BasicVector, MultibodyForces, CreateArrowOutputCalcCallback,
CreateArrowOutputAllocCallback, ArrowVisualization)
from pydrake.solvers import mathematicalprogram

class BoxController(LeafSystem):
  def __init__(self, sim_dt, robot_type, all_plant, robot_plant, mbw, kp, kd, robot_gv_kp, robot_gv_ki, robot_gv_kd, robot_instance, ball_instance, fully_actuated):
    LeafSystem.__init__(self)

    # Saves whether the entire system will be actuated.
    self.fully_actuated = fully_actuated

    # Save the robot type.
    self.set_name('box_controller')
    self.robot_type = robot_type

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

    # Initialize the embedded sim.
    self.embedded_sim = EmbeddedSim(sim_dt, kp, kd, robot_gv_kp, robot_gv_ki, robot_gv_kd, fully_actuated)

    # Set the controller type.
    self.controller_type = 'BlackboxDynamics'#'NoFrictionalForcesApplied'

    # Set the output size.
    self.command_output_size = self.robot_and_ball_plant.num_velocities()

    # Create contexts.
    self.mbw_context = mbw.CreateDefaultContext()
    self.robot_context = robot_plant.CreateDefaultContext()
    self.robot_and_ball_context = self.mbw.GetMutableSubsystemContext(
      self.robot_and_ball_plant, self.mbw_context)

    # Set PID gains.
    self.cartesian_kp = kp
    self.cartesian_kd = kd
    self.robot_gv_kp = robot_gv_kp
    self.robot_gv_ki = robot_gv_ki
    self.robot_gv_kd = robot_gv_kd

    # Set the control frequency.
    self.control_freq = 1000.0  # 1000 Hz.

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
    self._DeclareVectorOutputPort("command_output",
        BasicVector(self.command_output_size),
        self.DoControlCalc) # Output 0.

    # Get the geometry query input port.
    self.geometry_query_input_port = self.robot_and_ball_plant.get_geometry_query_input_port()

    # Set up ball c.o.m. acceleration visualization.
    self.ball_acceleration_visualization_output_port = self._DeclareAbstractOutputPort(
        "arrow_output",
        CreateArrowOutputAllocCallback(),
        CreateArrowOutputCalcCallback(self.OutputBallAccelerationAsGenericArrow))

    # Actuator limits.
    self.actuator_limit = float('inf')

    # TODO: delete this line.
    print 'WARNING: box_controller.py is violating the const System assumption'
    self.ball_accel_from_controller = np.array([0, 0, 0])

  # Gets the value of the integral term in the state.
  def get_integral_value(self, context):
    return context.get_continuous_state_vector().CopyToVector()

  # Sets the value of the integral term in the state.
  def set_integral_value(self, context, qint):
    assert len(qint) == self.nq_robot()
    context.get_mutable_continuous_state_vector().SetFromVector(qint)

  # Gets the ball body from the robot and ball plant.
  def get_ball_from_robot_and_ball_plant(self):
    return self.robot_and_ball_plant.GetBodyByName("ball")

  # Gets the foot links from the robot and ball plant.
  def get_foot_links_from_robot_and_ball_plant(self):
    return [ self.robot_and_ball_plant.GetBodyByName("box") ]

  # Gets the foot links from the robot tree.
  def get_foot_links_from_robot_plant(self):
    return [ self.robot_plant.GetBodyByName("box") ]

  # Gets the world body from the robot and ball tree.
  def get_ground_from_robot_and_ball_plant(self):
    return self.robot_and_ball_plant.GetBodyByName("ground_body")

  def get_input_port_estimated_robot_q(self):
    return self.get_input_port(self.input_port_index_estimated_robot_q)

  def get_input_port_estimated_robot_v(self):
    return self.get_input_port(self.input_port_index_estimated_robot_qd)

  def get_input_port_estimated_ball_q(self):
    return self.get_input_port(self.input_port_index_estimated_ball_q)

  def get_input_port_estimated_ball_v(self):
    return self.get_input_port(self.input_port_index_estimated_ball_v)

  def get_output_port_control(self):
    return self.get_output_port(0)

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

  # Gets the robot and ball configuration.
  def get_q_all(self, context):
    qrobot = self.get_q_robot(context)
    qball = self.get_q_ball(context)
    all_q = np.zeros([len(qrobot) + len(qball)])
    self.robot_and_ball_plant.SetPositionsInArray(self.robot_instance, qrobot, all_q)
    self.robot_and_ball_plant.SetPositionsInArray(self.ball_instance, qball, all_q)
    return all_q

  # Gets the robot and ball velocities.
  def get_v_all(self, context):
    vrobot = self.get_v_robot(context)
    vball = self.get_v_ball(context)
    all_v = np.zeros([len(vrobot) + len(vball)])
    self.robot_and_ball_plant.SetVelocitiesInArray(self.robot_instance, vrobot, all_v)
    self.robot_and_ball_plant.SetVelocitiesInArray(self.ball_instance, vball, all_v)
    return all_v

  # Gets the robot configuration.
  def get_q_robot(self, context):
    return self.EvalVectorInput(context, self.get_input_port_estimated_robot_q().get_index()).CopyToVector()

  # Gets the ball configuration.
  def get_q_ball(self, context):
    return self.EvalVectorInput(context, self.get_input_port_estimated_ball_q().get_index()).CopyToVector()

  # Gets the robot velocity.
  def get_v_robot(self, context):
    return self.EvalVectorInput(context, self.get_input_port_estimated_robot_v().get_index()).CopyToVector()

  # Gets the ball velocity
  def get_v_ball(self, context):
    return self.EvalVectorInput(context, self.get_input_port_estimated_ball_v().get_index()).CopyToVector()

  ### "Private" methods below.

  # Debugging function for visualizing the desired ball acceleration using
  # white arrows.
  def OutputBallAccelerationAsGenericArrow(self, controller_context):
    # Get the desired ball acceleration.
    vdot_ball_des = self.plan.GetBallQVAndVdot(controller_context.get_time())[-self.nv_ball():]
    vdot_ball_des = np.reshape(vdot_ball_des, [self.nv_ball(), 1])

    # Get the translational ball acceleration.
    xdd_ball_des = np.reshape(vdot_ball_des[3:6], [-1])

    # Evaluate the ball center-of-mass.
    all_plant = self.robot_and_ball_plant
    ball_body = self.get_ball_from_robot_and_ball_plant()
    X_WB = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, ball_body)
    com = X_WB.translation()

    # Populate the arrow visualization data structure.
    arrow_viz = ArrowVisualization()
    arrow_viz.origin_W = com
    arrow_viz.target_W = com + xdd_ball_des
    arrow_viz.color_rgb = np.array([1, 1, 1])  # White.

    # TODO: Delete this.
    # Construct a second one for the computed ball acceleration.
    arrow_viz_2 = ArrowVisualization()
    arrow_viz_2.origin_W = com
    arrow_viz_2.target_W = com + self.ball_accel_from_controller
    arrow_viz_2.color_rgb = np.array([1, 0, 1])
    return [ arrow_viz, arrow_viz_2 ]

    # A list must be returned.
    return [ arrow_viz ]

  # Makes a sorted pair.
  def MakeSortedPair(self, a, b):
    if b > a:
      return (b, a)
    else:
      return (a, b)

  # Loads all plans into the controller.
  def LoadPlans(self):
    from pydrake.common import FindResourceOrThrow

    # Set the Drake prefix.
    prefix = 'drake/examples/iiwa_soccer/'
    path = 'plan_box_curve/'

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

    if self.robot_type == 'box':
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

  # Constructs the Jacobian matrices.
  def ConstructJacobians(self, contacts, q):

    # Get the tree.
    all_plant = self.robot_and_ball_plant

    # Get the numbers of contacts and generalized velocities.
    nc = len(contacts)

    # Set the number of generalized velocities.
    nv = all_plant.num_velocities()

    # Size the matrices.
    N = np.empty([nc, nv])
    S = np.empty([nc, nv])
    T = np.empty([nc, nv])
    Ndot_v = np.empty([nc, 1])
    Sdot_v = np.empty([nc, 1])
    Tdot_v = np.empty([nc, 1])

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    self.UpdateRobotAndBallConfigurationForGeometricQueries(q)
    query_object = all_plant.EvalAbstractInput(
      self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
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
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)

      # The reported point on A's surface (A) in the world frame (W).
      pr_WA = point_pair.p_WCa

      # The reported point on B's surface (B) in the world frame (W).
      pr_WB = point_pair.p_WCb

      # Get the point of contact in the world frame.
      pc_W = (pr_WA + pr_WB) * 0.5

      # Get the geometric Jacobian for the velocity of the contact point
      # as moving with Body A.
      J_WAc = all_plant.CalcPointsGeometricJacobianExpressedInWorld(
          self.robot_and_ball_context, body_A.body_frame(), pr_WA)

      # Get the geometric Jacobian for the velocity of the contact point
      # as moving with Body B.
      J_WBc = all_plant.CalcPointsGeometricJacobianExpressedInWorld(
          self.robot_and_ball_context, body_B.body_frame(), pr_WB)

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
      N[i,:] = n_BA_W.T.dot(J)
      S[i,:] = t1_BA_W.T.dot(J)
      T[i,:] = t2_BA_W.T.dot(J)

      # TODO: Set Ndot_v, Sdot_v, Tdot_v properly.
      Ndot_v *= 0 # = n_BA_W.T * Jdot_v
      Sdot_v *= 0 # = t1_BA_W.T * Jdot_v
      Tdot_v *= 0 # = t2_BA_W.T * Jdot_v

    return [ N, S, T, Ndot_v, Sdot_v, Tdot_v ]

  # Computes the control torques when contact is not desired.
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
    x = self.robot_plant.tree().get_mutable_multibody_state_vector(
      robot_context)
    assert len(x) == len(q_robot) + len(qd_robot)
    x[0:len(q_robot)-1] = q_robot
    x[-len(qd_robot):] = qd_robot

    # Get the generalized inertia matrix.
    M = self.robot_plant.tree().CalcMassMatrixViaInverseDynamics(robot_context)

    # Compute the contribution from force elements.
    robot_tree = self.robot_plant.tree()
    link_wrenches = MultibodyForces(robot_tree)

    # Compute the external forces.
    fext = -robot_tree.CalcInverseDynamics(
        robot_context, np.zeros([nv_robot(), 1]), link_wrenches)

    # Compute inverse dynamics.
    u = M * qddot - fext
    return u

  # Updates the robot and ball configuration in the relevant context so that
  # geometric queries can be performed in configuration q.
  def UpdateRobotAndBallConfigurationForGeometricQueries(self, q):

    # Set the state.
    self.robot_and_ball_plant.SetPositions(self.robot_and_ball_context, q)

  # Gets the signed distance between the ball and the ground.
  def GetSignedDistanceFromBallToGround(self, context):
    all_plant = self.robot_and_ball_plant

    # Get the ball body and foot bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    world_body = self.get_ground_from_robot_and_ball_plant()

    # Make sorted pair to check.
    ball_world_pair = self.MakeSortedPair(ball_body, world_body)

    # Get the current configuration of the robot and the foot.
    q = self.get_q_all(context)

    # Update the context to use configuration q1 in the query. This will modify
    # the mbw context, used immediately below.
    self.UpdateRobotAndBallConfigurationForGeometricQueries(q)

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
      self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    # Get the closest points on the robot foot and the ball corresponding to q1
    # and v0.
    closest_points = query_object.ComputeSignedDistancePairwiseClosestPoints()
    assert len(closest_points) > 0

    dist = 1e20
    for i in range(len(closest_points)):
      geometry_A_id = closest_points[i].id_A
      geometry_B_id = closest_points[i].id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair != ball_world_pair:
        continue
      dist = min(dist, closest_points[i].distance)

    return dist

  # Gets the signed distance between the ball and the foot.
  def GetSignedDistanceFromRobotToBall(self, context):
    all_plant = self.robot_and_ball_plant

    # Get the ball body and foot bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    foot_bodies = self.get_foot_links_from_robot_and_ball_plant()

    # Make sorted pairs to check.
    ball_foot_pairs = [0] * len(foot_bodies)
    for i in range(len(foot_bodies)):
      ball_foot_pairs[i] = self.MakeSortedPair(ball_body, foot_bodies[i])

    # Get the current configuration of the robot and the foot.
    q = self.get_q_all(context)

    # Update the context to use configuration q1 in the query. This will modify
    # the mbw context, used immediately below.
    self.UpdateRobotAndBallConfigurationForGeometricQueries(q)

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
      self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    # Get the closest points on the robot foot and the ball corresponding to q1
    # and v0.
    closest_points = query_object.ComputeSignedDistancePairwiseClosestPoints()
    assert len(closest_points) > 0

    dist = 1e20
    for i in range(len(closest_points)):
      geometry_A_id = closest_points[i].id_A
      geometry_B_id = closest_points[i].id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair not in ball_foot_pairs:
        continue
      dist = min(dist, closest_points[i].distance)

    return dist

  # Computes the control torques when contact is desired and the robot and the
  # ball are *not* in contact.
  def ComputeActuationForContactDesiredButNoContact(self, controller_context):
    # Get the relevant trees.
    all_plant = self.robot_and_ball_plant
    robot_plant = self.robot_plant

    # Get the generalized positions and velocities for the robot and the ball.
    q0 = self.get_q_all(controller_context)
    v0 = self.get_v_all(controller_context)

    # Set the state in the "all plant" context.
    all_plant.SetPositions(self.robot_and_ball_context, q0)
    all_plant.SetVelocities(self.robot_and_ball_context, v0)

    # Set the joint velocities for the robot to zero.
    self.robot_and_ball_plant.SetVelocities(self.robot_and_ball_context, self.robot_instance, np.zeros([self.nv_robot()]))

    # Transform the velocities to time derivatives of generalized
    # coordinates.
    qdot0 = self.robot_and_ball_plant.MapVelocityToQDot(self.robot_and_ball_context, v0)
    dt = 1.0/self.control_freq

    # Get the estimated position of the ball and the robot at the next time
    # step using a first order approximation to position and the current
    # velocities.
    q1 = q0 + dt * qdot0

    # Update the context to use configuration q1 in the query. This will modify
    # the mbw context, used immediately below.
    self.UpdateRobotAndBallConfigurationForGeometricQueries(q1)

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
        self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    # Get the robot and the ball bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    foot_bodies = self.get_foot_links_from_robot_and_ball_plant()
    foots_and_ball = [0] * len(foot_bodies)
    for i in range(len(foot_bodies)):
      foots_and_ball[i] = self.MakeSortedPair(ball_body, foot_bodies[i])

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
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)
      bodies = self.MakeSortedPair(body_A, body_B)

      # If the two bodies correspond to the foot and the ball, mark the
      # found index and stop looping.
      if bodies in foots_and_ball:
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
    body_A = all_plant.GetBodyFromFrameId(frame_A_id)
    body_B = all_plant.GetBodyFromFrameId(frame_B_id)
    if body_B != ball_body:
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
    X_wa = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, body_A)
    X_wb = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, body_B)
    closest_Aw = X_wa.multiply(closest_Aa)
    closest_Bw = X_wb.multiply(closest_Bb)

    # Get the vector from the closest point on the foot to the closest point
    # on the ball in the body frames.
    linear_v_des = (closest_Bw - closest_Aw) / dt

    # Get the robot current generalized position and velocity.
    q_robot = self.get_q_robot(controller_context)
    v_robot = self.get_v_robot(controller_context)

    # Set the state in the robot context to q_robot and qd_robot.
    x = robot_plant.GetMutablePositionsAndVelocities(self.robot_context)
    assert len(x) == len(q_robot) + len(v_robot)
    x[0:len(q_robot)] = q_robot
    x[-len(v_robot):] = v_robot

    # Get the geometric Jacobian for the velocity of the closest point on the
    # robot as moving with the robot Body A.
    foot_bodies_in_robot_plant = self.get_foot_links_from_robot_plant()
    for body in foot_bodies_in_robot_plant:
      if body.name() == body_A.name():
        foot_body_to_use = body
    J_WAc = self.robot_plant.CalcPointsGeometricJacobianExpressedInWorld(
        self.robot_context, foot_body_to_use.body_frame(), closest_Aw)
    q_robot_des = q_robot

    # Use resolved-motion rate control to determine the robot velocity that
    # would be necessary to realize the desired end-effector velocity.
    v_robot_des, residuals, rank, singular_values = np.linalg.lstsq(J_WAc, linear_v_des)
    dvdot = np.reshape(v_robot_des - v_robot, (-1, 1))

    # Set vdot_robot_des using purely error feedback.
    if self.nq_robot() != self.nv_robot():
      # Since the coordinates and velocities are different, we convert the
      # difference in configuration to a difference in velocity coordinates.
      # TODO: Explain why this is allowable.
      self.robot_plant.SetPositions(self.robot_context, q_robot)
      dv = np.reshape(self.robot_plant.MapQDotToVelocity(self.robot_context, q_robot_des - q_robot), (-1, 1))
      vdot = np.diag(np.reshape(self.robot_gv_kp, (-1))).dot(dv) + np.diag(np.reshape(self.robot_gv_kd, (-1))).dot(dvdot)
    else:
      vdot = np.diag(self.robot_gv_kp).dot(q_robot_des - q_robot) + np.diag(self.robot_gv_kd).dot(np.reshape(v_robot_des - v_robot), (-1, 1))

    # Get the generalized inertia matrix.
    M = robot_plant.CalcMassMatrixViaInverseDynamics(self.robot_context)

    # Compute the contribution from force elements.
    link_wrenches = MultibodyForces(robot_plant)

    # Compute the external forces.
    fext = np.reshape(-robot_plant.CalcInverseDynamics(
        self.robot_context, np.zeros([self.nv_robot()]), link_wrenches), (-1, 1))

    # Compute inverse dynamics.
    return M.dot(vdot) - fext

  # Constructs the robot actuation matrix.
  def ConstructRobotActuationMatrix(self):
    # We assume that each degree of freedom is actuatable. There is no way to
    # verify this because we want to be able to use free bodies as "robots" too.

    # First zero out the generalized velocities for the whole multibody.
    v = np.zeros([self.nv_robot() + self.nv_ball()])

    # Now set the velocities in the generalized velocity array to ones.
    ones_nv_robot = np.ones([self.nv_robot()])
    self.robot_and_ball_plant.SetVelocitiesInArray(self.robot_instance, ones_nv_robot, v)

    # The matrix is of size nv_robot() + nv_ball() x nv_robot().
    if self.fully_actuated == True:
      # _Everything_ is actuated.
      #return np.eye(self.nv_robot() + self.nv_ball())

      # Full size actuation matrix but only ball is actuated.
      v[:] *= 0
      self.robot_and_ball_plant.SetVelocitiesInArray(self.ball_instance, ones_nv_robot, v)
      return np.diag(v)

      # Full size actuation matrix but only robot is actuated.
      return np.diag(v)
    else:
      # Only the robot is actuated.
      B = np.zeros([self.nv_robot() + self.nv_ball(), self.nv_robot()])
      col_index = 0
      for i in range(self.nv_robot() + self.nv_ball()):
        if abs(v[i]) > 0.5:
          B[i, col_index] = 1
          col_index += 1

      return B

  # Constructs the matrix that zeros angular velocities for the ball (and
  # does not change the linear velocities).
  def ConstructBallVelocityWeightingMatrix(self):
    # Get the indices of generalized velocity that correspond to the ball.
    nv = self.nv_ball() + self.nv_robot()
    v = np.zeros([nv])
    dummy_ball_v = np.ones([self.nv_ball()])
    self.robot_and_ball_plant.SetVelocitiesInArray(self.ball_instance, dummy_ball_v, v)

    # Set the velocities weighting.
    W = np.zeros([len(dummy_ball_v), nv])
    vweighting_ball = np.zeros([self.nv_ball()])
    print 'NOTE: assigning weighting to angular velocities'
    #vweighting_ball[-3:] = np.ones([3])    # Last three components correspond to linear velocities (see spatial_velocity.h)
    vweighting_ball[-6:] = np.array([10, 10, 10, 1, 1, 1])
    vball_index = 0
    for i in range(nv):
      if abs(v[i]) > 0.5:
        W[vball_index, i] = vweighting_ball[vball_index]
        vball_index += 1

    return W

  # Computes the motor torques for ComputeActuationForContactDesiredAndContacting()
  # using the no-slip model.
  # iM: the inverse of the joint robot/ball generalized inertia matrix
  # fext: the generalized external forces acting on the robot/ball
  # vdot_ball_des: the desired spatial acceleration on the ball
  # Z: the contact normal/tan1/tan2 Jacobian matrix (all normal rows come first,
  #    all first-tangent direction rows come next, all second-tangent direction
  #    rows come last).
  # Zdot_v: the time derivative of the contact Jacobian matrices times the
  #         generalized velocities.
  # RETURNS: a tuple containing (1) the actuation forces, (2) the contact force
  #          magnitudes (along the contact normals, the first tangent direction,
  #          and the second contact tangent direction, respectively), and (3)
  #          the primal solution to the quadratic program.
  def ComputeContactControlMotorTorquesNoSlip(self, iM, fext, vdot_ball_des, Z, Zdot_v):
    # Construct the actuation and weighting matrices.
    B = self.ConstructRobotActuationMatrix()
    P = self.ConstructBallVelocityWeightingMatrix()

    # Primal variables are motor torques and contact force magnitudes.
    nv, nu = B.shape
    ncontact_variables = len(Zdot_v)
    nc = ncontact_variables/3
    nprimal = nu + ncontact_variables

    # Construct the matrices necessary to construct the Hessian.
    Z_rows, Z_cols = Z.shape
    D = np.zeros([nv, nprimal])
    D[0:nv, 0:nu] = B
    D[-Z_cols:, -Z_rows:] = Z.T

    # Set the Hessian matrix for the QP.
    H = D.T.dot(iM.dot(P.T).dot(P.dot(iM.dot(D))))

    # Verify that the Hessian is positive semi-definite.
    H = H + np.eye(H.shape[0]) * 1e-8
    np.linalg.cholesky(H)

    # Compute the linear terms.
    c = D.T.dot(iM.dot(P.T).dot(-vdot_ball_des + P.dot(iM.dot(fext))))

    # Set the affine constraint matrix.
    A = np.zeros([nc*4, nprimal])
    A[0:nc*3, :] = Z.dot(iM.dot(D))
    A[nc*3:, nu:nu+nc] = np.eye(nc)    # Constraint normal forces to be non-negative.
    b = np.zeros([nc*4, 1])
    b[0:nc*3] = -Z.dot(iM.dot(fext)) - np.reshape(Zdot_v, (-1, 1))

    # Solve the QP.
    prog = mathematicalprogram.MathematicalProgram()
    vars = prog.NewContinuousVariables(len(c), "vars")
    prog.AddQuadraticCost(H, c, vars)
    prog.AddLinearConstraint(A, b, np.ones([len(b), 1]) * 1e8, vars)
    result = prog.Solve()
    assert result == mathematicalprogram.SolutionResult.kSolutionFound
    z = prog.GetSolution(vars)

    # Get the actuation forces and the contact forces.
    f_act = z[0:nu]
    f_contact = z[nu:nprimal]

    # Get the normal forces and ensure that they are not tensile.
    f_contact_n = f_contact[0:nc]
    assert np.min(f_contact_n) >= -1e-8

    return [f_act, f_contact, z[0:nprimal], D, P, B]

  # Computes the applied forces when the ball is fully actuated.
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
    vdot_ball_des = np.reshape(vdot_ball_des, [self.nv_ball(), 1])

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

    # Set maximum number of loop iterations.
    max_loop_iterations = 100

    for i in range(max_loop_iterations):
        # Compute b.
        b = fext + epsilon

        # Solve the linear system M*[vdot_des_ball; 0] - Bu = fext + epsilon,
        # meaning that we want the robot to remain stationary.
        vdot_des = np.zeros([nv, 1])
        all_plant.SetVelocitiesInArray(self.ball_instance, vdot_ball_des, vdot_des)
        [u, resid, rank, singular_values] = np.linalg.lstsq(B, M.dot(vdot_des) - fext - epsilon)
        u = np.reshape(u, [-1])
        z = np.zeros(nprimal)
        z[0:nv] = np.reshape(vdot_des, [-1])
        z[:-nv] = u

        # Update the state in the embedded simulation.
        self.embedded_sim.UpdateTime(controller_context.get_time())
        self.embedded_sim.UpdatePlantPositions(q)
        self.embedded_sim.UpdatePlantVelocities(v)

        # Apply the controls to the embedded simulation.
        self.embedded_sim.ApplyControls(B.dot(u))

        # Simulate the system forward in time.
        self.embedded_sim.StepEmbeddedSimulation()

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
      print 'WARNING: BlackBoxDynamics controller did not terminate!'

    self.ball_accel_from_controller = all_plant.GetVelocitiesFromArray(self.ball_instance, z[0:nv])[-3:]
    print 'Motor torques: ' + str(u)
    print 'Ball velocity: ' + str(all_plant.GetVelocitiesFromArray(self.ball_instance, v))

    return [u, z[0:nv], z[0:nprimal], P, B, epsilon]

  # Computes the motor torques for ComputeActuationForContactDesiredAndContacting()
  # using a "learned" dynamics model.
  def ComputeContactControlMotorTorquesUsingLearnedDynamics(self, controller_context, M, fext, vdot_ball_des):
    # Construct the actuation and weighting matrices.
    B = self.ConstructRobotActuationMatrix()
    P = self.ConstructBallVelocityWeightingMatrix()

    # Primal variables are motor torques and accelerations.
    nv, nu = B.shape
    nprimal = nu + nv

    # Initialize epsilon.
    epsilon = np.zeros([nv, 1])

    # Construct the Hessian matrix and linear term.
    H = np.zeros([nprimal, nprimal])
    H[0:nv,0:nv] = P.T.dot(P)
    H[nv:,nv:] = np.eye(nu)
    c = np.zeros([nprimal, 1])
    c[0:nv] = -P.T.dot(vdot_ball_des)

    # Construct the equality constraint matrix (for the QP) corresponding to:
    # M\dot{v} - Bu = fext + epsilon
    A = np.zeros([nv, nv + nu])
    A[:,0:nv] = M
    A[:,nv:] = -B

    # Get the current system positions and velocities.
    q = self.get_q_all(controller_context)
    v = self.get_v_all(controller_context)

    # Set maximum number of loop iterations.
    max_loop_iterations = 100

    for i in range(max_loop_iterations):
        # Compute b.
        b = fext + epsilon

        # Solve the QP.
        prog = mathematicalprogram.MathematicalProgram()
        vars = prog.NewContinuousVariables(len(c), "vars")
        prog.AddQuadraticCost(H, c, vars)
        prog.AddLinearConstraint(A, b, b, vars)
        result = prog.Solve()
        assert result == mathematicalprogram.SolutionResult.kSolutionFound
        z = prog.GetSolution(vars)
        u = z[:-nv]

        # Update the state in the embedded simulation.
        self.embedded_sim.UpdateTime(controller_context.get_time())
        self.embedded_sim.UpdatePlantPositions(q)
        self.embedded_sim.UpdatePlantVelocities(v)

        # Apply the controls to the embedded simulation.
        self.embedded_sim.ApplyControls(B.dot(u))

        # Simulate the system forward in time.
        self.embedded_sim.StepEmbeddedSimulation()

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

        # Question: how do we get this controller to realize *exactly* the
        # desired ball accelerations at the beginning? What are the responsible
        # constraints if the desired ball accelerations *can't* be achieved?

    if i == max_loop_iterations - 1:
      print 'WARNING: BlackBoxDynamics controller did not terminate!'
    '''
    print 'External forces and actuator forces: ' + str(-fext - B.dot(np.reshape(u, (-1, 1))))
    print 'Contact forces: ' + str(epsilon)
    print 'Forces on the ball: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, -fext - B.dot(np.reshape(u, (-1, 1))) - epsilon)[-3:])
    print 'predicted ball acceleration: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, z[0:nv]))
    '''

    for i in range(len(u)):
      if u[i] > self.actuator_limit:
        u[i] = self.actuator_limit
      else:
        if u[i] < -self.actuator_limit:
          u[i] = -self.actuator_limit
    z = np.reshape(z, [-1, 1])
    print 'objective: ' + str(0.5 * z.T.dot(H.dot(z) + c))
    print 'desired ball acceleration: ' + str(vdot_ball_des.T)
    print 'ball acceleration from vdot_approx: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx))

    self.ball_accel_from_controller = self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, z[0:nv])[-3:]

    return [u, z[0:nv], z[0:nprimal], P, B, epsilon]

  # Computes the motor torques for ComputeActuationForContactDesiredAndContacting()
  # under the requirement that no tangential forces are applied- the robot
  # can only apply "normal" forces.
  # iM: the inverse of the joint robot/ball generalized inertia matrix
  # fext: the generalized external forces acting on the robot/ball
  # vdot_ball_des: the desired spatial acceleration on the ball
  # N: the contact normal Jacobian matrix.
  # Ndot_v: the time derivative of the contact normal Jacobian matrix times the
  #         generalized velocities.
  # RETURNS: a tuple containing (1) the actuation forces, (2) the contact force
  #          magnitudes (along the contact normals), and (3) the primal solution
  #          to the quadratic program.
  def ComputeContactControlMotorTorquesNoFrictionalForces(self, iM, fext, vdot_ball_des, N, Ndot_v):
    # Construct the actuation and weighting matrices.
    B = self.ConstructRobotActuationMatrix()
    P = self.ConstructBallVelocityWeightingMatrix()

    # Primal variables are motor torques and contact force magnitudes.
    B_rows, B_cols = B.shape
    nc = len(Ndot_v)
    nprimal = B_cols + nc
    nv = B_rows

    # Construct the matrices necessary to construct the Hessian.
    N_rows, N_cols = N.shape
    D = np.zeros([nv, nprimal])
    D[0:B_rows, 0:B_cols] = B
    D[-N_cols:, -N_rows:] = N.T

    # Set the Hessian matrix for the QP.
    H = D.T.dot(iM.dot(P.T).dot(P.dot(iM.dot(D))))

    # Verify that the Hessian is positive semi-definite.
    H = H + np.eye(H.shape[0]) * 1e-8
    np.linalg.cholesky(H)

    # Compute the linear terms.
    c = D.T.dot(iM.dot(P.T).dot(-vdot_ball_des + P.dot(iM.dot(fext))))

    # Set the affine constraint matrix.
    A = np.zeros([nc*2, nprimal])
    b = np.zeros([nc*2, 1])
    A[0:nc,:] = N.dot(iM.dot(D))
    A[nc:, -nc:] = np.eye(nc)
    b[0:nc] = -N.dot(iM.dot(fext)) - Ndot_v

    # Solve the QP.
    prog = mathematicalprogram.MathematicalProgram()
    vars = prog.NewContinuousVariables(len(c), "vars")
    prog.AddQuadraticCost(H, c, vars)
    prog.AddLinearConstraint(A, b, np.ones([len(b), 1]) * 1e8, vars)
    result = prog.Solve()
    assert result == mathematicalprogram.SolutionResult.kSolutionFound
    z = prog.GetSolution(vars)

    # Get the actuation forces and the contact forces.
    f_act = z[0:B_cols]
    f_contact = z[B_cols:nprimal]

    # Get the normal forces and ensure that they are not tensile.
    f_contact_n = f_contact[0:nc]
    assert np.min(f_contact_n) >= -1e-8

    return [f_act, f_contact, z[0:nprimal], D, P, B]

  # Computes the control torques when contact is desired and the robot and the
  # ball are in contact.
  def ComputeActuationForContactDesiredAndContacting(self, controller_context, contacts):
    # Alias the plant and its context.
    all_plant = self.robot_and_ball_plant
    all_context = self.robot_and_ball_context

    # Get the number of generalized positions and velocities.
    nv = all_plant.num_velocities()
    assert nv == self.nv_robot() + self.nv_ball()

    # Get the generalized positions and velocities.
    q = self.get_q_all(controller_context)
    v = self.get_v_all(controller_context)

    # Set the state in the "all plant" context.
    all_plant.SetPositions(all_context, q)
    all_plant.SetVelocities(all_context, v)

    # Get the generalized inertia matrix of the ball/robot system and compute
    # its Cholesky factorization.
    M = all_plant.CalcMassMatrixViaInverseDynamics(all_context)
    iM = np.linalg.inv(M)

    # Compute the contribution from force elements.
    link_wrenches = MultibodyForces(all_plant)
    all_plant.CalcForceElementsContribution(all_context, link_wrenches)

    # Compute the external forces.
    fext = -all_plant.CalcInverseDynamics(all_context, np.zeros([len(v)]), link_wrenches)
    fext = np.reshape(fext, [len(v), 1])

    # Get the desired ball acceleration.
    vdot_ball_des = self.plan.GetBallQVAndVdot(controller_context.get_time())[-self.nv_ball():]
    vdot_ball_des = np.reshape(vdot_ball_des, [self.nv_ball(), 1])

    # Get the Jacobians at the point of contact: N, S, T, and construct Z and
    # Zdot_v.
    nc = len(contacts)
    N, S, T, Ndot_v, Sdot_v, Tdot_v = self.ConstructJacobians(contacts, q)
    Z = np.zeros([N.shape[0] * 3, N.shape[1]])
    Z[0:nc,:] = N
    Z[nc:2*nc,:] = S
    Z[-nc:,:] = T

    # Set the time-derivatives of the Jacobians times the velocity.
    Zdot_v = np.zeros([nc * 3])
    Zdot_v[0:nc] = Ndot_v[:,0]
    Zdot_v[nc:2*nc] = Sdot_v[:, 0]
    Zdot_v[-nc:] = Tdot_v[:, 0]

    # Compute torques without applying any tangential forces.
    if self.controller_type == 'NoFrictionalForcesApplied':
      f_act, f_contact, zprimal, D, P, B = self.ComputeContactControlMotorTorquesNoFrictionalForces(iM, fext, vdot_ball_des, N, Ndot_v)
    if self.controller_type == 'NoSlip':
      f_act, f_contact, zprimal, D, P, B = self.ComputeContactControlMotorTorquesNoSlip(iM, fext, vdot_ball_des, Z, Zdot_v)
    if self.controller_type == 'BlackboxDynamics':
      f_act, f_contact, zprimal, P, B, f_contact_generalized = self.ComputeContactControlMotorTorquesUsingLearnedDynamics(controller_context, M, fext, vdot_ball_des)

    if 0:
      print nc
      print f_contact.shape
      print N.shape
      print zprimal.shape

    # Compute the generalized contact forces.
    f_contact_generalized = None
    if self.controller_type == 'NoFrictionalForcesApplied':
      f_contact_generalized = N.T.dot(f_contact)
    if self.controller_type == 'NoSlip':
      f_contact_generalized = Z.T.dot(f_contact)

    # Output logging information.
    if 0:
        vdot = iM.dot(D.dot(zprimal) + fext)
        P_vdot = P.dot(vdot)
        print "N * v: " + str(N.dot(v))
        print "S * v: " + str(S.dot(v))
        print "T * v: " + str(T.dot(v))
        print "Ndot * v: " + str(Ndot_v)
        print "Zdot * v: " + str(Zdot_v)
        print "fext: " + str(fext)
        print "M: "
        print M
        print "P: "
        print P
        print "D: "
        print D
        print "B: "
        print B
        print "N: "
        print N
        print "Z: "
        print Z
        print "contact forces: " + str(f_contact)
        print "vdot: " + str(vdot)
        print "vdot (desired): " + str(vdot_ball_des)
        print "P * vdot: " + str(P_vdot)
        print "torque: " + str(f_act)

    return [f_act, f_contact_generalized ]

  # Finds contacts only between the ball and the robot.
  def FindBallGroundContacts(self, all_q):
    # Get contacts between the robot and ball, and ball and the ground.
    contacts = self.FindContacts(all_q)

    # Get the ball and ground bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    world_body = self.get_ground_from_robot_and_ball_plant()

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
      self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    # Get the tree corresponding to all bodies.
    all_plant = self.robot_and_ball_plant

    # Remove contacts between all but the ball and the ground.
    i = 0
    while i < len(contacts):
      geometry_A_id = contacts[i].id_A
      geometry_B_id = contacts[i].id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair != self.MakeSortedPair(ball_body, world_body):
        contacts[i] = contacts[-1]
        del contacts[-1]
      else:
        i += 1

    return contacts

  # Finds contacts only between the ball and the robot.
  def FindRobotBallContacts(self, all_q):
    # Get contacts between the robot and ball, and ball and the ground.
    contacts = self.FindContacts(all_q)

    # Get the ball body and foot bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    foot_bodies = self.get_foot_links_from_robot_and_ball_plant()

    # Make sorted pairs to check.
    ball_foot_pairs = [0] * len(foot_bodies)
    for i in range(len(foot_bodies)):
      ball_foot_pairs[i] = self.MakeSortedPair(ball_body, foot_bodies[i])

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
      self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    # Get the tree corresponding to all bodies.
    all_plant = self.robot_and_ball_plant

    # Remove contacts between all but the robot foot and the ball and the
    # ball and the ground.
    i = 0
    while i < len(contacts):
      geometry_A_id = contacts[i].id_A
      geometry_B_id = contacts[i].id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair not in ball_foot_pairs:
        contacts[i] = contacts[-1]
        del contacts[-1]
      else:
        i += 1

    return contacts

  # Gets the vector of contacts.
  def FindContacts(self, all_q):
    # Set q in the context.
    self.UpdateRobotAndBallConfigurationForGeometricQueries(all_q)

    # Get the tree corresponding to all bodies.
    all_plant = self.robot_and_ball_plant

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
        self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    # Determine the set of contacts.
    contacts = query_object.ComputePointPairPenetration()

    # Get the ball body and foot bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    foot_bodies = self.get_foot_links_from_robot_and_ball_plant()
    world_body = self.get_ground_from_robot_and_ball_plant()

    # Make sorted pairs to check.
    ball_foot_pairs = [0] * len(foot_bodies)
    for i in range(len(foot_bodies)):
      ball_foot_pairs[i] = self.MakeSortedPair(ball_body, foot_bodies[i])
    ball_world_pair = self.MakeSortedPair(ball_body, world_body)

    # Remove contacts between all but the robot foot and the ball and the
    # ball and the ground.
    i = 0
    while i < len(contacts):
      geometry_A_id = contacts[i].id_A
      geometry_B_id = contacts[i].id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = all_plant.GetBodyFromFrameId(frame_A_id)
      body_B = all_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair not in ball_foot_pairs and body_A_B_pair != ball_world_pair:
        contacts[i] = contacts[-1]
        del contacts[-1]
      else:
        i += 1

    return contacts

  # Determines whether the ball and the ground are in contact.
  def IsBallContactingGround(self, contacts):
    # Get the ball and ground bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    ground_body = self.get_ground_from_robot_and_ball_plant()

    # Make sorted pairs to check.
    ball_ground_pair = self.MakeSortedPair(ball_body, ground_body)

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
      self.robot_and_ball_context,
      self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    for contact in contacts:
      geometry_A_id = contact.id_A
      geometry_B_id = contact.id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = self.robot_and_ball_plant.GetBodyFromFrameId(frame_A_id)
      body_B = self.robot_and_ball_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair == ball_ground_pair:
        return True

    # No contact found.
    return False

  # Determines whether the ball and the robot are in contact.
  def IsRobotContactingBall(self, contacts):
    # Get the ball body and foot bodies.
    ball_body = self.get_ball_from_robot_and_ball_plant()
    foot_bodies = self.get_foot_links_from_robot_and_ball_plant()

    # Make sorted pairs to check.
    ball_foot_pairs = [0] * len(foot_bodies)
    for i in range(len(foot_bodies)):
      ball_foot_pairs[i] = self.MakeSortedPair(ball_body, foot_bodies[i])

    # Evaluate scene graph's output port, getting a SceneGraph reference.
    query_object = self.robot_and_ball_plant.EvalAbstractInput(
      self.robot_and_ball_context,
      self.geometry_query_input_port.get_index()).get_value()
    inspector = query_object.inspector()

    for contact in contacts:
      geometry_A_id = contact.id_A
      geometry_B_id = contact.id_B
      frame_A_id = inspector.GetFrameId(geometry_A_id)
      frame_B_id = inspector.GetFrameId(geometry_B_id)
      body_A = self.robot_and_ball_plant.GetBodyFromFrameId(frame_A_id)
      body_B = self.robot_and_ball_plant.GetBodyFromFrameId(frame_B_id)
      body_A_B_pair = self.MakeSortedPair(body_A, body_B)
      if body_A_B_pair in ball_foot_pairs:
        return True

    # No contact found.
    return False

  # Calculate what torques to apply to the joints.
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
            # Find contacts.
            contacts = self.FindContacts(q)

            # Two cases: in the first, the robot and the ball are already in contact,
            # as desired. In the second, the robot desires to be in contact, but the
            # ball and robot are not contacting: the robot must intercept the ball.
            if self.IsRobotContactingBall(contacts):
                print 'Contact desired and contact detected at time ' + str(context.get_time())
                tau, f_contact_generalized = self.ComputeActuationForContactDesiredAndContacting(context, contacts)
            else:
                print 'Contact desired and no contact detected at time ' + str(context.get_time())
                tau = self.ComputeActuationForContactDesiredButNoContact(context)
        else:
            # No contact desired.
            print 'Contact not desired at time ' + str(context.get_time())
            tau = self.ComputeActuationForContactNotDesired(context)


    # Set the torque output.
    mutable_torque_out = output.get_mutable_value()
    mutable_torque_out[:] = np.zeros(mutable_torque_out.shape)
    return
    if self.fully_actuated:
        mutable_torque_out[:] = torque.flatten()
    else:
        self.robot_and_ball_plant.SetVelocitiesInArray(self.robot_instance, tau.flatten(), mutable_torque_out)

  def _DoCalcTimeDerivatives(self, context, derivatives):
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


#  def DoPublish(context, publish_events):
