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
          if not self.controller.IsRobotContactingBall(q, contacts):
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
      old_dist = self.controller.GetSignedDistanceFromRobotToBall()

      # "Simulate" a small change to the box by computing a first-order
      # discretization to the new velocity and then using that to compute a first-order
      # discretization to the new configuration.
      dt = 1e-3
      vnew_robot = v_robot + dt*vdot_robot
      qd_robot = self.robot_plant.MapVelocityToQDot(robot_context, vnew_robot)
      qnew_robot = q_robot + dt*qd_robot

      # Get the new distance from the box to the ball.
      self.controller.SetPositions(self.controller.robot_and_ball_context, qnew_robot)
      new_dist = self.controller.GetSignedDistanceFromRobotToBall()
      logging.debug('Old distance: ' + str(old_dist) + ' new distance: ' +  str(new_dist))

      self.assertLess(new_dist, old_dist)

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

      # Primal variables are motor forces and accelerations.
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

      # Primal variables are motor forces and accelerations.
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
        [Z, Zdot_v] = self.controller.SetZAndZdot_v(N, S, T, Ndot_v, Sdot_v, Tdot_v)

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
        return [fz, Z, Zdot_v, iM, fext]


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
        if not self.controller.IsRobotContactingBall(q, contacts):
          t += dt
          continue

        logging.info('-- test_NoSeparationControllerContactForcesConsistent() - identified testable time/state at '
                    't=' + str(t))
        self.PrintContacts(q)

        # Get the desired acceleration.
        vdot_ball_des = plan.GetBallQVAndVdot(self.controller_context.get_time())[-self.controller.nv_ball():]
        vdot_ball_des = np.reshape(vdot_ball_des, [self.controller.nv_ball(), 1])[-3:]
        logging.debug('desired ball linear acceleration: ' + str(vdot_ball_des))

        # Get the Jacobians.
        N, S, T, Ndot_v, Sdot_v, Tdot_v = self.controller.ConstructJacobians(contacts, q, v)
        nc = N.shape[0]
        ball_ground_contact_index = self.controller.GetBallGroundContactIndex(q, contacts)
        S_ground = S[ball_ground_contact_index, :]
        T_ground = T[ball_ground_contact_index, :]
        Sdot_v_ground = Sdot_v[ball_ground_contact_index]
        Tdot_v_ground = Tdot_v[ball_ground_contact_index]

        # Verify that the velocities in each direction of the contact frame are zero.
        Nv = N.dot(v)
        Sv = S.dot(v)
        Tv = T.dot(v)
        self.assertLess(np.linalg.norm(Nv), zero_velocity_tol)
        self.assertLess(np.linalg.norm(Sv), zero_velocity_tol)
        self.assertLess(np.linalg.norm(Tv), zero_velocity_tol)

        # Run the no-slip controller.
        dummy, Z, Zdot_v, iM, fext = self.ComputeContactForcesWithoutControl(q, v, N, S, T, Ndot_v, Sdot_v, Tdot_v)
        P = self.controller.ConstructBallVelocityWeightingMatrix()
        B = self.controller.ConstructRobotActuationMatrix()
        u, fz = self.controller.ComputeContactControlMotorForcesNoSlip(iM, fext, vdot_ball_des, Z, Zdot_v, N, Ndot_v,
            S_ground, Sdot_v_ground, T_ground, Tdot_v_ground)
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
