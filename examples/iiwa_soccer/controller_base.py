import math
import scipy.optimize
import numpy as np
import logging
from recordclass import recordclass
from embedded_box_soccer_sim import EmbeddedSim

from pydrake.all import (LeafSystem, ComputeBasisFromAxis, PortDataType, BasicVector, MultibodyForces,
SpatialVelocity, ComputePoseDiffInCommonFrame, DoDifferentialInverseKinematics,
CreateArrowOutputCalcCallback, CreateArrowOutputAllocCallback, ArrowVisualization)
from pydrake.solvers import mathematicalprogram

# General controller components (not specific to, e.g., the box controller).
class ControllerBase(LeafSystem):

    def __init__(self, sim_dt, penetration_allowance, robot_type, all_plant, robot_plant, mbw, robot_instance,
            ball_instance):
        LeafSystem.__init__(self)

        # Save the robot type.
        self.robot_type = robot_type

        # Save the robot and ball instances.
        self.robot_instance = robot_instance
        self.ball_instance = ball_instance

        # Get the plants.
        self.robot_plant = robot_plant
        self.robot_and_ball_plant = all_plant
        self.mbw = mbw

        # Initialize the embedded sim.
        self.embedded_sim = EmbeddedSim(sim_dt, penetration_allowance)

        # Set the output size.
        self.command_output_size = self.robot_and_ball_plant.num_velocities()

        # Create contexts.
        self.mbw_context = mbw.CreateDefaultContext()
        self.robot_context = robot_plant.CreateDefaultContext()
        self.robot_and_ball_context = self.mbw.GetMutableSubsystemContext(self.robot_and_ball_plant, self.mbw_context)

        # Save the initial configuration.
        self.q0 = self.robot_and_ball_plant.GetPositions(self.robot_and_ball_context)

        # Set the control frequency.
        self.control_freq = 1000.0  # 1000 Hz.

        # Declare states and ports.
        self.DeclareContinuousState(self.nq_robot())   # For integral control state.
        self.input_port_index_estimated_robot_q = self.DeclareInputPort(
                PortDataType.kVectorValued, self.nq_robot()).get_index()
        self.input_port_index_estimated_robot_qd = self.DeclareInputPort(
                PortDataType.kVectorValued, self.nv_robot()).get_index()
        self.input_port_index_estimated_ball_q = self.DeclareInputPort(
                PortDataType.kVectorValued, self.nq_ball()).get_index()
        self.input_port_index_estimated_ball_v = self.DeclareInputPort(
                PortDataType.kVectorValued, self.nv_ball()).get_index()
        self.DeclareVectorOutputPort("command_output",
                BasicVector(self.command_output_size),
                self.DoControlCalc) # Output 0.

        # Get the geometry query input port.
        self.geometry_query_input_port = self.robot_and_ball_plant.get_geometry_query_input_port()

        # Set up ball c.o.m. acceleration visualization.
        self.ball_acceleration_visualization_output_port = self.DeclareAbstractOutputPort(
                "arrow_output",
                CreateArrowOutputAllocCallback(),
                CreateArrowOutputCalcCallback(self.OutputBallAccelerationAsGenericArrow))

    # Function for constructing Z and Zdot_v from N, S, T, etc.
    def SetZAndZdot_v(self, N, S, T, Ndot_v, Sdot_v, Tdot_v):
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

    # Gets the number of robot actuators.
    def num_robot_actuators(self):
        if self.robot_plant.num_actuators() == 0:
            return self.robot_plant.num_velocities()
        else:
            return self.robot_plant.num_actuators()

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

    ### "Protected" methods below.

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

    # Constructs the Jacobian matrices.
    def ConstructJacobians(self, contacts, q, v):

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
        all_plant.SetPositions(self.robot_and_ball_context, q)
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

    # Gets the index of contact between the ball and the ground.
    def GetBallGroundContactIndex(self, q, contacts):
        # Get the ball and ground bodies.
        ball_body = self.get_ball_from_robot_and_ball_plant()
        world_body = self.get_ground_from_robot_and_ball_plant()

        # Ensure that the context is set correctly.
        all_plant = self.robot_and_ball_plant
        all_context = self.robot_and_ball_context
        all_plant.SetPositions(all_context, q)

        # Evaluate scene graph's output port, getting a SceneGraph reference.
        query_object = self.robot_and_ball_plant.EvalAbstractInput(
            self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
        inspector = query_object.inspector()

        # Get the tree corresponding to all bodies.
        all_plant = self.robot_and_ball_plant

        # Get the desired contact.
        ball_ground_contact_index = -1
        for i in range(len(contacts)):
            geometry_A_id = contacts[i].id_A
            geometry_B_id = contacts[i].id_B
            frame_A_id = inspector.GetFrameId(geometry_A_id)
            frame_B_id = inspector.GetFrameId(geometry_B_id)
            body_A = all_plant.GetBodyFromFrameId(frame_A_id)
            body_B = all_plant.GetBodyFromFrameId(frame_B_id)
            body_A_B_pair = self.MakeSortedPair(body_A, body_B)
            if body_A_B_pair == self.MakeSortedPair(ball_body, world_body):
                assert ball_ground_contact_index == -1
                ball_ground_contact_index = i

        return i

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
        self.robot_and_ball_plant.SetPositions(self.robot_and_ball_context, all_q)

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
    def IsBallContactingGround(self, contacts, q):
        # Get the ball and ground bodies.
        ball_body = self.get_ball_from_robot_and_ball_plant()
        ground_body = self.get_ground_from_robot_and_ball_plant()

        # Make sorted pairs to check.
        ball_ground_pair = self.MakeSortedPair(ball_body, ground_body)

        # Ensure that the context is set correctly.
        all_plant = self.robot_and_ball_plant
        all_context = self.robot_and_ball_context
        all_plant.SetPositions(all_context, q)

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
    def IsRobotContactingBall(self, contacts, q):
        # Get the ball body and foot bodies.
        ball_body = self.get_ball_from_robot_and_ball_plant()
        foot_bodies = self.get_foot_links_from_robot_and_ball_plant()

        # Ensure that the context is set correctly.
        all_plant = self.robot_and_ball_plant
        all_context = self.robot_and_ball_context
        all_plant.SetPositions(all_context, q)

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

    # Gets the signed distance between the ball and the ground.
    def GetSignedDistanceFromBallToGround(self):
        all_plant = self.robot_and_ball_plant

        # Get the ball body and foot bodies.
        ball_body = self.get_ball_from_robot_and_ball_plant()
        world_body = self.get_ground_from_robot_and_ball_plant()

        # Make sorted pair to check.
        ball_world_pair = self.MakeSortedPair(ball_body, world_body)

        # Evaluate scene graph's output port, getting a SceneGraph reference.
        query_object = self.robot_and_ball_plant.EvalAbstractInput(
            self.robot_and_ball_context, self.geometry_query_input_port.get_index()).get_value()
        inspector = query_object.inspector()

        # Get the closest points on the robot foot and the ball corresponding to q1
        # and v0.
        self.ResetSignedDistanceQuery()
        closest_points = query_object.ComputeSignedDistancePairwiseClosestPoints()
        assert len(closest_points) == 3

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
    def GetSignedDistanceFromRobotToBall(self):
        all_plant = self.robot_and_ball_plant

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

        # Get the closest points on the robot foot and the ball corresponding to q1
        # and v0.
        self.ResetSignedDistanceQuery()
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

    # Gets the desired interpenetration (a proxy for deformation) of the ball when it is being driven through contact.
    # Returns a non-positive number (in mm).
    def get_desired_ball_signed_distance_under_contact(self):
        return -2.5e-3

    '''
    Computes the motor forces that minimize deviation from the desired acceleration without using dynamics information.
    It uses a gradient-based optimization strategy. This controller uses the embedded simulator to compute contact
    forces, rather than attempting to predict the contact forces that the simulator will generate.

    NOTE: I have found this control strategy to be insufficient alone. It does not track the planned box motion, which
    means that perturbations cause errors to grow over time.
    '''
    def ComputeOptimalContactControlMotorForces(self, controller_context, q, v, vdot_des, weighting_type='ball-linear'):
        vdot_des = np.reshape(vdot_des, [-1, 1])

        P = self.ConstructVelocityWeightingMatrix(weighting_type)
        nu = self.ConstructRobotActuationMatrix().shape[1]

        # Get the current system positions and velocities.
        nv = len(v)

        # The objective function.
        def objective_function(u):
            vdot_approx = self.ComputeApproximateAcceleration(controller_context, q, v, u)
            delta = P.dot(vdot_approx - vdot_des)
            return np.linalg.norm(delta)

        #result = scipy.optimize.minimize(objective_function, np.random.normal(np.zeros([nu])))
        result = scipy.optimize.minimize(objective_function, np.array([0, 0, 0, 0, 0, 0]))
        logging.info('scipy.optimize success? ' + str(result.success))
        logging.info('scipy.optimize message: ' +  result.message)
        logging.info('scipy.optimize result: ' + str(result.x))
        logging.info('function evaluation: ' + str(objective_function(result.x)))
        u_best = result.x

        vdot_approx = self.ComputeApproximateAcceleration(controller_context, q, v, u_best)
        logging.info('ball desired acceleration: ' +
                str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, vdot_des)))
        logging.info('ball approximate acceleration: ' +
                str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)))
        logging.info('robot desired acceleration: ' +
                str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.robot_instance, vdot_des)))
        logging.info('robot approximate acceleration: ' +
                str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.robot_instance, vdot_approx)))
        logging.info('P * approximate acceleration: ' +
                str(P.dot(vdot_approx)))

        return u_best

    def ResetSignedDistanceQuery(self):
        all_plant = self.robot_and_ball_plant
        all_context = self.robot_and_ball_context
        qcurrent = all_plant.GetPositions(all_context)
        all_plant.SetPositions(all_context, self.q0)
        query_object = all_plant.EvalAbstractInput(all_context, self.geometry_query_input_port.get_index()).get_value()
        inspector = query_object.inspector()
        query_object.ComputeSignedDistancePairwiseClosestPoints()
        all_plant.SetPositions(self.robot_and_ball_context, qcurrent)


    # Gets a normal vector and the signed distance between the ball and the foot.
    # Returns a data structure holding three elements:
    #    normal_foot_ball_W a unit vector pointing from a witness point on the foot body to a witness point on the ball
    #                       body, expressed in the global frame.
    #    phi the signed distance between the foot body and the ball body
    #    closest_foot_body the Body of the foot that yields the phi closest to negative infinity.
    def GetNormalAndSignedDistanceFromRobotToBall(self, q0):
        all_plant = self.robot_and_ball_plant

        # Set the configuration in the all-plant context.
        all_plant.SetPositions(self.robot_and_ball_context, q0)

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

        # Initialize the normal/signed distance data structure.
        normal_and_signed_distance = recordclass(
                'NormalAndSignedDistanceData', 'phi normal_foot_ball_W closest_foot_body')

        # Get the closest points on the robot foot and the ball corresponding to q1
        # and v0.
        self.ResetSignedDistanceQuery()
        closest_points = query_object.ComputeSignedDistancePairwiseClosestPoints()
        assert len(closest_points) > 0

        # NOTE: This calculation only produces the needed data when the minimum
        # distance is greater than zero.
        phi = 1e20
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
            if closest_points[i].distance < phi:
                X_WA = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, body_A)
                X_WB = all_plant.EvalBodyPoseInWorld(self.robot_and_ball_context, body_B)
                phi = closest_points[i].distance
                min_dist_bodies = body_A_B_pair
                p_ACw = X_WA.multiply(closest_points[i].p_ACa)
                p_BCw = X_WB.multiply(closest_points[i].p_BCb)
                normal_foot_ball_W = p_ACw - p_BCw
                normal_nrm = np.linalg.norm(normal_foot_ball_W)
                if normal_nrm < 1e-15:
                    # This must be a kissing configuration. Make the normal oppose the direction from the ball to the
                    # point on the ball.
                    if body_A is ball_body:
                        v_ballC_W = p_ACw - X_WA.translation()
                    else:
                        v_ballC_W = p_BCw - X_WB.translation()
                    v_ballC_W /= np.linalg.norm(v_ballC_W)
                    normal_foot_ball_W = -v_ballC_W
                else:
                    normal_foot_ball_W /= normal_nrm

                    # The normal currently points from Body B to Body A. Reverse the normal if Body A does not correspond
                    # to the ball.
                    if not min_dist_bodies[0] is ball_body:
                        normal_foot_ball_W = -normal_foot_ball_W

        # Now check the minimum distance for positivity.
        if phi >= 0:

            normal_and_signed_distance.phi = phi
            normal_and_signed_distance.normal_foot_ball_W = normal_foot_ball_W
            if min_dist_bodies[0] is ball_body:
                normal_and_signed_distance.closest_foot_body = min_dist_bodies[1]
            else:
                normal_and_signed_distance.closest_foot_body = min_dist_bodies[0]
        else:
            # Do almost exactly the same operation but with "contacts" instead of signed distances. Why must this be so
            # hard?
            contacts = self.FindContacts(q0)
            assert len(contacts) > 0

            phi = 1e20
            for i in range(len(contacts)):
                geometry_A_id = contacts[i].id_A
                geometry_B_id = contacts[i].id_B
                frame_A_id = inspector.GetFrameId(geometry_A_id)
                frame_B_id = inspector.GetFrameId(geometry_B_id)
                body_A = all_plant.GetBodyFromFrameId(frame_A_id)
                body_B = all_plant.GetBodyFromFrameId(frame_B_id)
                body_A_B_pair = self.MakeSortedPair(body_A, body_B)
                if body_A_B_pair not in ball_foot_pairs:
                    continue
                if -contacts[i].depth < phi:
                    phi = -contacts[i].depth
                    min_dist_bodies = body_A_B_pair
                    normal_foot_ball_W = contacts[i].nhat_BA_W

                    # The normal currently points from Body B to Body A. Reverse the normal if Body A does not correspond
                    # to the ball.
                    if not min_dist_bodies[0] is ball_body:
                        normal_foot_ball_W = -normal_foot_ball_W

            normal_and_signed_distance.phi = phi
            normal_and_signed_distance.normal_foot_ball_W = normal_foot_ball_W
            if min_dist_bodies[0] is ball_body:
                normal_and_signed_distance.closest_foot_body = min_dist_bodies[1]
            else:
                normal_and_signed_distance.closest_foot_body = min_dist_bodies[0]

        return normal_and_signed_distance

    ''' TODO: Delete me.
    # Converts a skew symmetric tensor to a three-dimensional vector.
    def Antiskew(M):
        v = np.zeros([3])
        v[0] = M[1, 2] - M[2, 1]
        v[1] = M[2, 0] - M[0, 2]
        v[2] = M[0, 1] - M[1, 0]
        return v * 0.5

    # Converts the difference between two poses to a spatial velocity vector.
    def PoseDifferential(self, X_WA, X_WB)

        # Get the relative pose from X_WA to X_WB: X_AB.
        X_AB = X_WA.inverse() * X_WB

        # Translational components are simple.
        translational = X_AB.translation()

        # X_AB represents \dot{R}. X_WA represents R. Therefore, the angular
        # velocity \omega is antiskew(\dot{R} * inv(R)), from the relationship
        # \dot{R} = skew(\omega) * R.
        rotational = self.Antiskew(X_AB.linear() * X_WA.linear())

        return SpatialVelocity(w=rotational, v=translational)
    '''

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
        # Only the robot is actuated.
        B = np.zeros([self.nv_robot() + self.nv_ball(), self.nv_robot()])
        col_index = 0
        for i in range(self.nv_robot() + self.nv_ball()):
            if abs(v[i]) > 0.5:
                B[i, col_index] = 1
                col_index += 1

        return B

    # Constructs the matrix that weights certain velocities.
    #  weighting_type: 'ball-linear' (everything but the ball linear velocities are given zero weight),
    #                  'ball-full' (ball linear and angular velocities are specially weighted), or
    #                  'full'      (robot and ball linear and angular velocities are weighted).
    # Returns a nv x nv-sized matrix, where nv is the number of velocity variables in the plant.
    def ConstructVelocityWeightingMatrix(self, weighting_type='ball-linear'):
        ball_radius = 0.1
        box_length = 0.4
        box_width = 0.4
        box_depth = 0.04
        box_radius = math.sqrt(box_length*box_length + box_width*box_width + box_depth*box_depth)

        assert self.robot_and_ball_plant.num_velocities(self.robot_instance) == 6
        if weighting_type == 'ball-linear':
            ball_v = np.array([0, 0, 0, 1, 1, 1])
            box_v = np.zeros(self.robot_and_ball_plant.num_velocities(self.robot_instance))
        elif weighting_type == 'ball-full':
            ball_v = np.array([ball_radius, ball_radius, ball_radius, 1, 1, 1])
            box_v = np.zeros(self.robot_and_ball_plant.num_velocities(self.robot_instance))
        elif weighting_type == 'full':
            ball_v = np.array([ball_radius, ball_radius, ball_radius, 1, 1, 1])
            box_v = np.array([box_radius, box_radius, box_radius, 1, 1, 1])
        else:
            assert False

        # Get the indices of generalized velocity that correspond to the ball.
        nv = self.nv_ball() + self.nv_robot()
        v = np.zeros([nv])
        self.robot_and_ball_plant.SetVelocitiesInArray(self.ball_instance, ball_v, v)
        self.robot_and_ball_plant.SetVelocitiesInArray(self.robot_instance, box_v, v)

        # Set the velocities weighting.
        return np.diag(v)

    # Does RMRC for a desired velocity. The desired velocity can be either a 3-dimensional vector, in which case it
    # refers to a translational velocity expressed in the world frame, or a 6-dimensional vector, in which case it
    # corresponds to a spatial velocity vector (angular velocity on top, translational velocity on bottom),
    # expressed in the world frame.
    def RMRC(self, robot_plant, robot_context, v_W, frame):
        p_BoFo_B = np.zeros([3, 1])    # Origin of the given frame.
        if v_W.size == 3:
            J_W = robot_plant.CalcPointsGeometricJacobianExpressedInWorld(
                    robot_context, frame, p_BoFo_B)
        else:
            assert v_W.size == 6
            J_W = robot_plant.CalcFrameGeometricJacobianExpressedInWorld(
                    robot_context, frame_B=frame, p_BoFo_B=p_BoFo_B)
        v_W.resize([v_W.shape[0], 1])
        return np.linalg.lstsq(J_W, v_W)[0]  # First element is the solution.

    # Computes the approximate acceleration from a `q`, a `v`, and a `u` by running the simulation forward in time by
    # `dt`.
    def ComputeApproximateAcceleration(self, controller_context, q, v, u, dt=-1):
        # Update the step size, if necessary.
        if dt > 0:
            self.embedded_sim.delta_t = dt

        # Update the state in the embedded simulation.
        self.embedded_sim.UpdateTime(controller_context.get_time())
        self.embedded_sim.UpdatePlantPositions(q)
        self.embedded_sim.UpdatePlantVelocities(v)

        # Apply the controls to the embedded simulation and step the simulation
        # forward in time.
        B = self.ConstructRobotActuationMatrix()
        self.embedded_sim.ApplyControls(B.dot(u))
        #return self.embedded_sim.CalcAccelerations()
        self.embedded_sim.Step()

        # Get the new system velocity.
        vnew = self.embedded_sim.GetPlantVelocities()

        # Compute the estimated acceleration.
        return np.reshape((vnew - v) / self.embedded_sim.delta_t, (-1, 1))

