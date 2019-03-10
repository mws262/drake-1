  # Computes the control forces when contact is desired and the robot and the
  # ball are *not* in contact.
  def ComputeActuationForContactDesiredButNoContact(self, controller_context):
      # Get the relevant plants.
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

  # Computes the motor forces for ComputeActuationForContactDesiredAndContacting() using the no-slip contact model
  # *between the ball and the ground only*. Specifically, this function optimizes:
  # argmin vdot_sub_des 1/2 * (vdot_sub_des - P * vdot)' * W *
  #                             (vdot_sub_des - P * vdot)
  # subject to:
  # Gn * vdot_sub_des + dotGn * v_sub_des >= 0
  # Gs_ball * vdot_sub_des + dotGs_ball * v_sub_des = 0
  # Gt_ball * vdot_sub_des + dotGt_ball * v_sub_des = 0
  # M * vdot = fext + Gn' * fn + Gs' * fs + Gt' * ft + B * u
  # fn >= 0
  #
  # and preconditions:
  # Nv = 0
  #
  # iM: the inverse of the joint robot/ball generalized inertia matrix
  # fext: the generalized external forces acting on the robot/ball
  # vdot_ball_des: the desired spatial acceleration on the ball
  # Z: the contact normal/tan1/tan2 Jacobian matrix (all normal rows come first, all first-tangent direction rows come
  # next, all second-tangent direction rows come last). In the notation above, Z = [ Gn; Gs; Gt ].
  # Zdot_v: the time derivative of the contact Jacobian matrices times the generalized velocities. In the notation
  # above, Zdot = [ dotGn; dotGs; dotGt ].
  #
  # Returns: a tuple containing (1) the actuation forces (u), (2) the contact force magnitudes fz (along the contact
  # normals, the first tangent direction, and the second contact tangent direction, respectively),
  def ComputeContactControlMotorForcesNoSlip(self, iM, fext, vdot_ball_des, Z, Zdot_v, N, Ndot_v, S_ground,
          Sdot_v_ground, T_ground, Tdot_v_ground):
      # Construct the actuation and weighting matrices.
      B = self.ConstructRobotActuationMatrix()
      P = self.ConstructBallVelocityWeightingMatrix()

      # Primal variables are motor forces and contact force magnitudes.
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

      # Compute the linear terms.
      c = D.T.dot(iM.dot(P.T).dot(-vdot_ball_des + P.dot(iM.dot(fext))))

      # Set the affine constraint matrix. There are nc + 1 constraints:
      # normal accelerations are zero (nc constraints),
      # no slip at the ground (2 constraints).
      A = np.zeros([nc + 2, nprimal])
      A[0:nc, :] = N.dot(iM.dot(D))
      A[nc, :] = S_ground.dot(iM.dot(D))
      A[nc+1, :] = T_ground.dot(iM.dot(D))
      b = np.zeros([nc+2, 1])
      b[0:nc] = -N.dot(iM.dot(fext)) - np.reshape(Ndot_v, (-1, 1))
      b[nc] = -S_ground.dot(iM.dot(fext)) - np.reshape(Sdot_v_ground, (-1, 1))
      b[nc+1] = -T_ground.dot(iM.dot(fext)) - np.reshape(Tdot_v_ground, (-1, 1))

      # Add a compressive-force constraint on the normal contact forces.
      prog = mathematicalprogram.MathematicalProgram()
      vars = prog.NewContinuousVariables(len(c), "vars")
      lb = np.zeros(nprimal)
      lb[0:nu] = np.ones(nu) * -float('inf')
      lb[nu + nc:] = np.ones(nc * 2) * -float('inf')
      ub = np.ones(nprimal) * float('inf')
      prog.AddBoundingBoxConstraint(lb, ub, vars)

      # Solve the QP.
      prog.AddQuadraticCost(H, c, vars)
      prog.AddLinearConstraint(A, b, np.ones([len(b), 1]) * 1e8, vars)
      result = prog.Solve()
      assert result == mathematicalprogram.SolutionResult.kSolutionFound
      z = prog.GetSolution(vars)

      # Get the actuation forces and the contact forces.
      u = np.reshape(z[0:nu], [-1, 1])
      fz = np.reshape(z[nu:nprimal], [-1, 1])

      # Determine the friction coefficient for each point of contact.
      for i in range(nc):
          fs = fz[i+nc]
          ft = fz[i+nc*2]
          tan_force = math.sqrt(fs*fs + ft*ft)
          logging.info('Forces for contact ' + str(i) + ' normal: ' + str(fz[i]) + '  tangent: ' + str(tan_force))
          if tan_force < 1e-8:
              logging.info('Friction coefficient for contact ' + str(i) + ' unknown')
          else:
              logging.info('Friction coefficient for contact ' + str(i) + ' = ' + str(tan_force/fz[i]))

      # Get the normal forces and ensure that they are not tensile.
      f_contact_n = fz[0:nc]
      assert np.min(f_contact_n) >= -1e-8

      return [u, fz]

  # Computes the control forces when contact is desired and the robot and the
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

      # Get the Jacobians at the point of contact: N, S, T, and construct Z and
      # Zdot_v.
      nc = len(contacts)
      N, S, T, Ndot_v, Sdot_v, Tdot_v = self.ConstructJacobians(contacts, q, v)
      Z = np.zeros([N.shape[0] * 3, N.shape[1]])
      Z[0:nc,:] = N
      Z[nc:2*nc,:] = S
      Z[-nc:,:] = T

      # Get the Jacobians for ball/ground contact.
      ball_ground_contact_index = self.GetBallGroundContactIndex(q, contacts)
      S_ground = S[ball_ground_contact_index, :]
      T_ground = T[ball_ground_contact_index, :]
      Sdot_v_ground = Sdot_v[ball_ground_contact_index]
      Tdot_v_ground = Tdot_v[ball_ground_contact_index]

      # Set the time-derivatives of the Jacobians times the velocity.
      Zdot_v = np.zeros([nc * 3])
      Zdot_v[0:nc] = Ndot_v[:,0]
      Zdot_v[nc:2*nc] = Sdot_v[:, 0]
      Zdot_v[-nc:] = Tdot_v[:, 0]

      # Get the desired accelerations.
      t = controller_context.get_time()
      vdot_des = np.zeros([nv, 1])
      vdot_ball_des = self.plan.GetBallQVAndVdot(t)[-6:]
      vdot_box_des = self.plan.GetRobotQVAndVdot(t)[-6:]

      # Feed error feedback gains in for the robot.
      nq_robot = all_plant.num_positions(self.robot_instance)
      q_box = all_plant.GetPositionsFromArray(self.robot_instance, q)
      q_box_des = self.plan.GetRobotQVAndVdot(t)[0:nq_robot]
      nv_robot = all_plant.num_velocities(self.robot_instance)
      v_box = all_plant.GetVelocitiesFromArray(self.robot_instance, v)
      v_box_des = self.plan.GetRobotQVAndVdot(t)[nq_robot:nq_robot+nv_robot]

      # Modify the desired accelerations using error feedback gains.
      kp = 1e-5
      kd = 1e-4
      dt = 1.0/self.control_freq
      deltaq = np.zeros([all_plant.num_positions(), 1])
      all_plant.SetPositionsInArray(self.robot_instance, q_box_des - q_box, deltaq)
      deltav = all_plant.MapQDotToVelocity(all_context, deltaq)
      deltav_box_des = all_plant.GetVelocitiesFromArray(self.robot_instance, deltav)
      vdot_box_des += kp/(dt*dt) * deltav_box_des
      vdot_box_des += kd/dt * (v_box_des - v_box)

      # Set the desired acceleration now.
      all_plant.SetVelocitiesInArray(self.ball_instance, vdot_ball_des, vdot_des)
      all_plant.SetVelocitiesInArray(self.robot_instance, vdot_box_des, vdot_des)

      # Compute forces without applying any tangential forces.
      if self.controller_type == 'NoSlip':
          u, f_contact = self.ComputeContactControlMotorForcesNoSlip(iM, fext, vdot_ball_des, Z, Zdot_v, N, Ndot_v,
                  S_ground, Sdot_v_ground, T_ground, Tdot_v_ground)
      if self.controller_type == 'NoSeparation':
          u, f_contact = self.ComputeContactControlMotorForcesNoSeparation(iM, fext, vdot_ball_des, Z, N, Ndot_v)
      if self.controller_type == 'BlackBoxDynamics':
          u = self.ComputeOptimalContactControlMotorForces(controller_context, q, v, vdot_des, weighting_type='full')

      # Compute the generalized contact forces.
      f_contact_generalized = None
      if self.controller_type == 'NoSlip' or self.controller_type == 'NoSeparation':
          f_contact_generalized = Z.T.dot(f_contact)

      # Output logging information.
      '''
      vdot = iM.dot(D.dot(zprimal) + fext)
      P_vdot = P.dot(vdot)
      logging.debug("N * v: " + str(N.dot(v)))
      logging.debug("S * v: " + str(S.dot(v)))
      logging.debug("T * v: " + str(T.dot(v)))
      logging.debug("Ndot * v: " + str(Ndot_v))
      logging.debug("Zdot * v: " + str(Zdot_v))
      logging.debug("fext: " + str(fext))
      logging.debug("M: ")
      logging.debug(M)
      logging.debug("P: ")
      logging.debug(P)
      logging.debug("D: ")
      logging.debug(D)
      logging.debug("B: ")
      logging.debug(B)
      logging.debug("N: ")
      logging.debug(N)
      logging.debug("Z: ")
      logging.debug(Z)
      logging.debug("contact forces: " + str(f_contact))
      logging.debug("vdot: " + str(vdot))
      logging.debug("vdot (desired): " + str(vdot_ball_des))
      logging.debug("P * vdot: " + str(P_vdot))
      logging.debug("torque: " + str(f_act))
      '''

      return u


  # Computes the motor forces for ComputeActuationForContactDesiredAndContacting()
  # using a "learned" dynamics model.
  # See BoxController.ComputeContactControlMotorForcesNoSlip() for description of parameters.
  def ComputeContactControlMotorForcesUsingLearnedDynamics(self, controller_context, M, fext, vdot_ball_des, Z, Zdot_v):
    # Get the actuation matrix.
    B = self.ConstructRobotActuationMatrix()

    # Get the current system positions and velocities.
    q = self.get_q_all(controller_context)
    v = self.get_v_all(controller_context)
    nv = len(v)

    # Compute inverse(M)
    iM = np.linalg.inv(M)

    # Check N*v, S*v, T*v.
    logging.debug('Z*v: ' + str(Z.dot(v)))

    # Computes the reality gap.
    def reality_gap(epsilon):
      epsilon = np.reshape(epsilon, [-1, 1])

      # Call the no slip controller.
      [u, fz] = self.ComputeContactControlMotorForcesNoSlip(iM, fext + epsilon, vdot_ball_des, Z, Zdot_v)

      # Make u and fz column vectors.
      fz = np.reshape(fz, [-1, 1])
      u = np.reshape(u, [-1, 1])

      # Compute the approximate acceleration.
      vdot_approx = self.ComputeApproximateAcceleration(controller_context, q, v, u)

      # Compute the difference between the dynamics computed by the no slip controller and the true dynamics.
      # The dynamics will be:
      # M * vdot = fext + Gn' * fn + Gs' * fs + Gt' * ft + B * u + epsilon.
      # The "reality gap" is the unaccounted for force.
      delta = M.dot(vdot_approx) - fext - Z.T.dot(fz) - B.dot(u) - epsilon
      logging.debug('reality gap: ' + str(np.reshape(delta, -1)))
      return np.reshape(delta, -1)

    # Attempt to solve the nonlinear system of equations.
    epsilon = np.reshape(scipy.optimize.fsolve(reality_gap, np.zeros([nv, 1])), [-1, 1])

    # Call the no slip controller.
    [u, fz] = self.ComputeContactControlMotorForcesNoSlip(iM, fext + epsilon, vdot_ball_des, Z, Zdot_v)

    # Compute the estimated acceleration.
    vdot_approx = self.ComputeApproximateAcceleration(controller_context, q, v, u)

    logging.warning('Residual error norm: ' + str(np.linalg.norm(reality_gap(epsilon))))
    logging.debug('External forces and actuator forces: ' + str(-fext - B.dot(np.reshape(u, (-1, 1)))))
    logging.debug('Spatial contact force acting at the center-of-mass of the robot: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.robot_instance, Z.T.dot(fz))))
    logging.debug('Spatial contact force acting at the center-of-mass of the ball: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, Z.T.dot(fz))))
    logging.debug('Unmodeled forces: ' + str(epsilon))
    logging.debug('Forces on the ball: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, -fext - B.dot(np.reshape(u, (-1, 1))) - epsilon)[-3:]))
    logging.debug('desired ball acceleration: ' + str(vdot_ball_des.T))
    logging.debug('ball acceleration from vdot_approx: ' + str(self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)))

    self.ball_accel_from_controller = self.robot_and_ball_plant.GetVelocitiesFromArray(self.ball_instance, vdot_approx)[-3:]

    return [u, fz]

  # Computes the motor forces for ComputeActuationForContactDesiredAndContacting() using the no-separation contact
  # model. Specifically, this function optimizes:
  # argmin vdot_sub_des 1/2 * (vdot_sub_des - P * vdot)' * W *
  #                             (vdot_sub_des - P * vdot)
  # subject to:
  # Gn * vdot_sub_des + dotGn * v_sub_des >= 0
  # M * vdot = fext + Gn' * fn + Gs' * fs + Gt' * ft + B * u
  # fn >= 0
  #
  # and preconditions:
  # Nv = 0
  #
  # iM: the inverse of the joint robot/ball generalized inertia matrix
  # fext: the generalized external forces acting on the robot/ball
  # vdot_ball_des: the desired spatial acceleration on the ball
  # Z: the contact normal/tan1/tan2 Jacobian matrix (all normal rows come first, all first-tangent direction rows come
  #    next, all second-tangent direction rows come last). In the notation above, Z = [ Gn; Gs; Gt ].
  # N: the contact normal Jacobian matrix.
  # Ndot_v: the time derivative of the normal contact Jacobian matrix times the generalized velocities.
  #
  # Returns: a tuple containing (1) the actuation forces (u), (2) the contact force magnitudes fz (along the contact
  # normals, the first tangent direction, and the second contact tangent direction, respectively),
  def ComputeContactControlMotorForcesNoSeparation(self, iM, fext, vdot_ball_des, Z, N, Ndot_v):
    # Construct the actuation and weighting matrices.
    B = self.ConstructRobotActuationMatrix()
    P = self.ConstructBallVelocityWeightingMatrix()

    # Primal variables are motor forces and contact force magnitudes.
    nv, nu = B.shape
    ncontact_variables = Z.shape[0]
    nc = N.shape[0]
    nprimal = nu + ncontact_variables

    # Construct the matrices necessary to construct the Hessian.
    Z_rows, Z_cols = Z.shape
    D = np.zeros([nv, nprimal])
    D[0:nv, 0:nu] = B
    D[-Z_cols:, -Z_rows:] = Z.T

    # Set the Hessian matrix for the QP.
    H = D.T.dot(iM.dot(P.T).dot(P.dot(iM.dot(D))))

    # Compute the linear terms.
    c = D.T.dot(iM.dot(P.T).dot(-vdot_ball_des + P.dot(iM.dot(fext))))

    # Set the affine constraint matrix.
    A = N.dot(iM.dot(D))
    b = -N.dot(iM.dot(fext)) - np.reshape(Ndot_v, (-1, 1))

    # Formulate the QP.
    prog = mathematicalprogram.MathematicalProgram()
    vars = prog.NewContinuousVariables(len(c), "vars")
    prog.AddQuadraticCost(H, c, vars)
    prog.AddLinearConstraint(A, b, b, vars)

    # Add a compressive-force constraint on the normal contact forces.
    lb = np.zeros(nprimal)
    lb[0:nu] = np.ones(nu) * -float('inf')
    lb[nu + nc:] = np.ones(nc * 2) * -float('inf')
    ub = np.ones(nprimal) * float('inf')
    prog.AddBoundingBoxConstraint(lb, ub, vars)

    # Solve the QP.
    result = prog.Solve()
    assert result == mathematicalprogram.SolutionResult.kSolutionFound
    z = prog.GetSolution(vars)

    # Get the actuation forces and the contact forces.
    u = np.reshape(z[0:nu], [-1, 1])
    fz = np.reshape(z[nu:nprimal], [-1, 1])

    # Get the normal forces and ensure that they are not tensile.
    fz_n = fz[0:nc]
    assert np.min(fz_n) >= -1e-8

    # Verify that the normal acceleration constraint is met.
    vdot = iM.dot(Z.T.dot(fz) + B.dot(u) + fext)
    assert np.linalg.norm(N.dot(vdot) + Ndot_v) < 1e-2
    logging.info('Objective function value: ' + str(np.linalg.norm(P.dot(vdot) - vdot_ball_des)))

    # Determine the friction coefficient for each point of contact.
    for i in range(nc):
      fs = fz[i+nc]
      ft = fz[i+nc*2]
      tan_force = math.sqrt(fs*fs + ft*ft)
      logging.info('Forces for contact ' + str(i) + ' normal: ' + str(fz[i]) + '  tangent: ' + str(tan_force))
      if tan_force < 1e-8:
          logging.info('Friction coefficient for contact ' + str(i) + ' unknown')
      else:
          logging.info('Friction coefficient for contact ' + str(i) + ' = ' + str(tan_force/fz[i]))

    return [u, fz]

  # Computes the motor forces for ComputeActuationForContactDesiredAndContacting() that minimize deviation from the
  # desired acceleration using no dynamics information and a gradient-free optimization strategy.
  # This controller uses the simulator to compute contact forces, rather than attempting to predict the contact forces
  # that the simulator will generate.
  def ComputeOptimalContactControlMotorForcesDerivativeFree(self, controller_context, q, v, vdot_ball_des):

    P = self.controller.ConstructBallVelocityWeightingMatrix()
    nu = self.controller.ConstructRobotActuationMatrix().shape[1]

    # Get the current system positions and velocities.
    nv = len(v)

    # The objective function.
    def objective_function(u):
      vdot_approx = self.controller.ComputeApproximateAcceleration(controller_context, q, v, u)
      delta = P.dot(vdot_approx) - vdot_ball_des
      return np.linalg.norm(delta)

    '''
    # Do CMA-ES.
    sigma = 0.1
    u_best = np.array([0, 0, 0, -2.3675, 0, 0])
    fbest = objective_function(u_best)
    for i in range(1):
      es = cma.CMAEvolutionStrategy(u_best, sigma)
      es.optimize(objective_function)
      if es.result.fbest < fbest:
        fbest = es.result.fbest
        u_best = es.result.xbest
    '''

    res = scipy.optimize.minimize(objective_function, np.random.normal(np.zeros([nu])))
    print res.success
    print res.message
    u_best = res.x

    return u_best

