import numpy as np

class ManipulationPlan:

  # First seven variables of the vector correspond to robot configuration;
  # the next seven variables of the vector correspond to joint velocity. The
  # sole scalar value corresponds to the designated time.
  self.q_qdot_qddot_robot = [(0.0, np.array([]))]

  # First three variables: com position; next four variables: quaternion
  # orientation (qw qx qy qz); next three variables: translational velocity
  # (expressed in the world frame); next three variables: angular velocity
  # (expressed in the world frame). The sole scalar value corresponds to the
  # designated time.
  self.q_v_vdot_ball = [(0.0, np.array([]))]

  # Times and indicators of contact / no contact desired between the ball and
  # the robot. `true` indicates contact desired.
  self.contact_desired = [(0.0, True)]

  # The planned point of contact between the robot and the ball in the world
  # frame (first three components), and the time derivative of that point
  # (next three components). The sole scalar value corresponds to the
  # designated time.
  self.contact_kinematics = [(0.0, np.array([]))]

  # Gets the final time of the plan.
  def end_time():
    end_time = 0.0;
    if len(q_qdot_qddot_robot) > 0:
      end_time = max(end_time, q_qdot_qddot_robot[-1][0])
    if len(q_v_vdot_ball) > 0:
      end_time = max(end_time, q_v_vdot_ball[-1][0])
    if len(contact_desired) > 0:
      assert end_time >= contact_desired[-1][0])
    if len(contact_kinematics) > 0:
      assert(end_time >= contact_kinematics[-1][0])
    return end_time

  # Returns whether it is desired that the robot and ball be in contact at
  # desired time (`t`).
  def IsContactDesired(t):
    return contact_desired[SearchBinary(t, contact_desired)][1]
  
  # Gets the ball q, v, and vdot at a particular point in time.
  def GetBallQVAndVdot(t):
    return q_v_vdot_ball[SearchBinary(t, q_v_vdot_ball)][1]

  # Gets the ball q, v, and vdot at a particular point in time.
  def GetRobotQQdotAndQddot(t):
    return q_qdot_qddot_robot[SearchBinary(t, q_qdot_qddot_robot)][1]

  # Gets the contact kinematics at a particular point in time.
  # First three components are the contact point location in the global frame
  # and the second three components are the time derivative of that location
  # in the global frame.
  def GetContactKinematics(t):
    return contact_kinematics[SearchBinary(t, contact_kinematics)][1]

  def ReadRobotQQdotAndQddot(timings_fname, q_fname, qd_fname, qdd_fname):
    kDim = 21;

    q_qdot_qddot_robot.clear();

    # Read in timings.
    in_timings = open(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
        t = in_timings_str[i]
        q_qdot_qddot_robot.Append((t, np.ones((kDim,1)) * float('nan')))
    in_timings.close()

    #  Read in joint angles.
    in_q = open(q_fname);
    in_q_str = in_q.read()
# TODO: continue me here....
    for (size_t i = 0; i < q_qdot_qddot_robot.size(); ++i) {
      for (int j = 0; j < kDim/3; ++j) {
        in_q >> q_qdot_qddot_robot[i].second[j];
        assert(!in_q.eof() && !in_q.fail());
      }
    }

    #  Make sure there is nothing else to read.
    double dummy;
    in_q >> dummy;
    assert(in_q.eof() || in_q.fail());
    in_q.close();

    #  Read in joint velocities.
    std::ifstream in_qd(qd_fname);
    for (size_t i = 0; i < q_qdot_qddot_robot.size(); ++i) {
      for (int j = 0; j < kDim/3; ++j) {
        in_qd >> q_qdot_qddot_robot[i].second[j + kDim/3];
        assert(!in_qd.eof() && !in_qd.fail());
      }
    }

    #  Make sure there is nothing else to read.
    in_qd >> dummy;
    assert(in_qd.eof() || in_qd.fail());
    in_qd.close();

    #  Read in joint velocities.
    std::ifstream in_qdd(qdd_fname);
    for (size_t i = 0; i < q_qdot_qddot_robot.size(); ++i) {
      for (int j = 0; j < kDim/3; ++j) {
        in_qdd >> q_qdot_qddot_robot[i].second[j + 2 * kDim/3];
        assert(!in_qdd.eof() && !in_qdd.fail());
      }
    }

    #  Make sure there is nothing else to read.
    in_qdd >> dummy;
    assert(in_qdd.eof() || in_qdd.fail());
    in_qdd.close();

    #  Make sure there are no NaN's.
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i)
      for (int j = 0; j < q_v_vdot_ball[i].second.size(); ++j)
        assert(!std::isnan(q_qdot_qddot_robot[i].second[j]));
  }

  def ReadContactPoint(timings_fname, cp_fname, cp_dot_fname):
    kLocationDim = 3;
    kContactPointVelocityOffset = 3;
    
    contact_kinematics.clear();

    #  Read in timings.
    in_timings = file.read(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
      t = in_timings_str[i]
      contact_kinematics.Append((t, np.ones((kLocationDim * 2,1)) * float('nan')))
    in_timings.close()

    #  Read in contact point location over time.
    in_x = file.read(cp_fname, 'r')
    for (size_t i = 0; i < contact_kinematics.size(); ++i) {
      for (int j = 0; j < kLocationDim; ++j) {
        in_x >> contact_kinematics[i].second[j];
        assert(!in_x.eof() && !in_x.fail());
      }
    }

    #  Make sure there is nothing else to read.
    double dummy;
    in_x >> dummy;
    assert(in_x.eof() || in_x.fail());
    in_x.close();
    
    #  Read in contact point velocity over time.
    in_xdot = file.read(cp_dot_fname, 'r')
    for (size_t i = 0; i < contact_kinematics.size(); ++i) {
      for (int j = 0; j < kLocationDim; ++j) {
        in_xdot >> contact_kinematics[i].second[j + kContactPointVelocityOffset];
        assert(!in_xdot.eof() && !in_xdot.fail());
      }
    }

    #  Make sure there is nothing else to read.
    in_xdot >> dummy;
    assert(in_xdot.eof() || in_xdot.fail());
    in_xdot.close();
  }

  def ReadBallQVAndVdot(timings_fname,
                        com_locations_fname,
                        quats_fname,
                        com_velocity_fname,
                        angular_velocity_fname,
                        com_accel_fname,
                        angular_accel_fname,
                        contact_indicator_fname):
    kStatePlusAccelDim = 19;

    q_v_vdot_ball.clear();

    #  Read in timings.
    in_timings = file.open(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
        t = in_timings_str[i]
        q_v_vdot_ball.Append((t, np.ones((kStatePlusAccelDim,1)) * float('nan')))
    in_timings.close()

    #  Read in com locations.
    in_x = file.open(com_locations_fname, 'r')
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_x >> q_v_vdot_ball[i].second[j];
        assert(!in_x.eof() && !in_x.fail());
      }
    }

    #  Make sure there is nothing else to read.
    double dummy;
    in_x >> dummy;
    assert(in_x.eof() || in_x.fail());
    in_x.close();

    #  Read in unit quaternions.
    double quat_tol = 1e-7;
    int kQuatOffset = 3;
    in_quat = file.read(quats_fname, 'r')
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      for (int j = 0; j < 4; ++j) {
        in_quat >> q_v_vdot_ball[i].second[j + kQuatOffset];
        assert(!in_quat.eof() && !in_quat.fail());
      }
      assert(std::abs(q_v_vdot_ball[i].second.segment(
          kQuatOffset, 4).norm() - 1.0) < quat_tol);
    }

    #  Make sure there is nothing else to read.
    in_quat >> dummy;
    assert(in_quat.eof() || in_quat.fail());
    in_quat.close();

    #  Read in translational velocities.
    int kVOffset = 7;
    in_v = file.read(com_velocity_fname, 'r')
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_v >> q_v_vdot_ball[i].second[j + kVOffset];
        assert(!in_v.eof() && !in_v.fail());
      }
    }

    #  Make sure there is nothing else to read.
    in_v >> dummy;
    assert(in_v.eof() || in_v.fail());
    in_v.close();

    #  Read in angular velocities.
    int kWOffset = 10;
    std::ifstream in_w(angular_velocity_fname);
    assert(!in_w.fail());
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_w >> q_v_vdot_ball[i].second[j + kWOffset];
        assert(!in_w.eof() && !in_w.fail());
      }
    }

    #  Make sure there is nothing else to read.
    in_w >> dummy;
    assert(in_w.eof() || in_w.fail());
    in_w.close();

    #  Read in translational acceleration.
    int kVDotOffset = 13;
    in_vdot = file.open(com_accel_fname, 'r')
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_vdot >> q_v_vdot_ball[i].second[j + kVDotOffset];
        assert(!in_vdot.eof() && !in_vdot.fail());
      }
    }

    #  Make sure there is nothing else to read.
    in_vdot >> dummy;
    assert(in_vdot.eof() || in_vdot.fail());
    in_vdot.close();
    
    #  Read in angular accelerations.
    int kAlphaOffset = 16;
    in_alpha = file.read(angular_accel_fname, 'r')
    assert(!in_alpha.fail());
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_alpha >> q_v_vdot_ball[i].second[j + kAlphaOffset];
        assert(!in_alpha.eof() && !in_alpha.fail());
      }
    }

    #  Make sure there are no NaN's.
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i)
      for (int j = 0; j < q_v_vdot_ball[i].second.size(); ++j)
        assert(!std::isnan(q_v_vdot_ball[i].second[j]));

    #  Make sure there is nothing else to read.
    in_alpha >> dummy;
    assert(in_alpha.eof() || in_alpha.fail());
    in_alpha.close();
    
    #  Read in the contact indicators.
    in_contact_indicator = file.read(contact_indicator_fname, 'r')
    for (size_t i = 0; i < q_v_vdot_ball.size(); ++i) {
      bool status;
      in_contact_indicator >> status;
      assert(!in_contact_indicator.eof() && !in_contact_indicator.fail());
      contact_desired_.emplace_back(
          std::make_pair(q_v_vdot_ball[i].first, status));
    }
    in_contact_indicator.close();
  }

  def SearchBinary(t, vec):
    int left = 0
    right = len(vec.size()-1)
    while right - left > 1:
      mid = (left + right)/2;
      if (t > vec[mid](0)):
        left = mid
      else:
        right = mid
    assert left != right
    return left
