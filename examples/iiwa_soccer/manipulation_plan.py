import numpy as np
import math

class ManipulationPlan:

  def __init__(self):
    # First seven variables of the vector correspond to robot configuration
    # the next seven variables of the vector correspond to joint velocity. The
    # sole scalar value corresponds to the designated time.
    self.q_v_vdot_robot = [(0.0, np.array([]))]

    # First three variables: com position next four variables: quaternion
    # orientation (qw qx qy qz) next three variables: translational velocity
    # (expressed in the world frame) next three variables: angular velocity
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
  def end_time(self):
    end_time = 0.0
    if len(self.q_v_vdot_robot) > 0:
      end_time = max(end_time, self.q_v_vdot_robot[-1][0])
    if len(self.q_v_vdot_ball) > 0:
      end_time = max(end_time, self.q_v_vdot_ball[-1][0])
    if len(self.contact_desired) > 0:
      assert end_time >= self.contact_desired[-1][0]
    if len(self.contact_kinematics) > 0:
      assert end_time >= self.contact_kinematics[-1][0]
    return end_time

  # Returns whether it is desired that the robot and ball be in contact at
  # desired time (`t`).
  def IsContactDesired(self, t):
    return self.contact_desired[self.SearchBinary(t, self.contact_desired)][1]
  
  # Gets the ball q, v, and vdot at a particular point in time.
  def GetBallQVAndVdot(self, t):
    return self.q_v_vdot_ball[self.SearchBinary(t, self.q_v_vdot_ball)][1]

  # Gets the ball q, v, and vdot at a particular point in time.
  def GetRobotQVAndVdot(self, t):
    return self.q_v_vdot_robot[self.SearchBinary(t, self.q_v_vdot_robot)][1]

  # Gets the contact kinematics at a particular point in time.
  # First three components are the contact point location in the global frame
  # and the second three components are the time derivative of that location
  # in the global frame.
  def GetContactKinematics(self, t):
    return self.contact_kinematics[self.SearchBinary(t, self.contact_kinematics)][1]

  def ReadIiwaRobotQVAndVdot(self, timings_fname, q_fname, qd_fname, qdd_fname):
    kDim = 21  # 7 joints x 3

    del self.q_v_vdot_robot[:]

    # Read in timings.
    in_timings = open(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
        t = float(in_timings_str[i])
        self.q_v_vdot_robot.append((t, np.ones((kDim)) * float('nan')))
    in_timings.close()

    #  Read in joint angles.
    in_q = open(q_fname, 'r')
    in_q_str = in_q.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(kDim/3):
        self.q_v_vdot_robot[i][1][j] = float(in_q_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.q_v_vdot_robot) * kDim / 3
    in_q.close()

    #  Read in joint velocities.
    in_qd = open(qd_fname, 'r')
    in_qd_str = in_qd.read().split()
    str_index = 0 
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(kDim/3):
        self.q_v_vdot_robot[i][1][j + kDim/3] = float(in_qd_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.q_v_vdot_robot) * kDim / 3
    in_qd.close()

    #  Read in joint velocities.
    in_qdd = open(qdd_fname, 'r')
    in_qdd_str = in_qdd.read().split()
    str_index = 0 
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(kDim/3):
        self.q_v_vdot_robot[i][1][j + 2*kDim/3] = float(in_qdd_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.q_v_vdot_robot) * kDim / 3
    in_qdd.close()

    #  Make sure there are no NaN's.
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(len(self.q_v_vdot_ball[i][1])):
        assert not math.isnan(self.q_v_vdot_robot[i][1][j])

  def ReadBoxRobotQVAndVdot(self, timings_fname,
                        com_locations_fname,
                        quats_fname,
                        com_velocity_fname,
                        angular_velocity_fname,
                        com_accel_fname,
                        angular_accel_fname):
    kStatePlusAccelDim = 19

    del self.q_v_vdot_robot[:]

    #  Read in timings.
    in_timings = open(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
      t = float(in_timings_str[i])
      self.q_v_vdot_robot.append((t, np.ones((kStatePlusAccelDim)) * float('nan')))
    in_timings.close()

    # Set offsets in the vector.
    kQuatOffset = 0
    kComOffset = 4
    kWOffset = 7
    kVOffset = 10
    kAlphaOffset = 13
    kVDotOffset = 16
    kEnd = kVDotOffset + 3

    #  Read in com locations.
    in_x = open(com_locations_fname, 'r')
    in_x_str = in_x.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(3):
        self.q_v_vdot_robot[i][1][j + kComOffset] = float(in_x_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.q_v_vdot_robot)*3
    in_x.close()

    #  Read in unit quaternions.
    quat_tol = 1e-7
    in_quat = open(quats_fname, 'r')
    in_quat_str = in_quat.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(4):
        self.q_v_vdot_robot[i][1][j + kQuatOffset] = float(in_quat_str[str_index])
        str_index = str_index + 1
      qw = self.q_v_vdot_robot[i][1][kQuatOffset+0]
      qx = self.q_v_vdot_robot[i][1][kQuatOffset+1]
      qy = self.q_v_vdot_robot[i][1][kQuatOffset+2]
      qz = self.q_v_vdot_robot[i][1][kQuatOffset+3]
      assert abs(qw*qw + qx*qx + qy*qy + qz*qz - 1) < quat_tol
      self.q_v_vdot_robot[i][1][kQuatOffset+0] = qw
      self.q_v_vdot_robot[i][1][kQuatOffset+1] = qx
      self.q_v_vdot_robot[i][1][kQuatOffset+2] = qy
      self.q_v_vdot_robot[i][1][kQuatOffset+3] = qz
    assert str_index == len(self.q_v_vdot_robot)*4
    in_quat.close()

    # TODO: replace this when Matt provides velocities and accelerations.
    for i in range(len(self.q_v_vdot_robot)):
      self.q_v_vdot_robot[i][1][kWOffset:] = np.zeros([kEnd - kWOffset])

    '''
    #  Read in translational velocities.
    in_v = open(com_velocity_fname, 'r')
    in_v_str = in_v.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(3):
        self.q_v_vdot_robot[i][1][j + kVOffset] = float(in_v_str[str_index])
        str_index = str_index + 1
    in_v.close()

    #  Read in angular velocities.
    in_w = open(angular_velocity_fname, 'r')
    in_w_str = in_w.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(3):
        self.q_v_vdot_robot[i][1][j + kWOffset] = float(in_w_str[str_index])
        str_index = str_index + 1
    in_w.close()

    #  Read in translational acceleration.
    in_vdot = open(com_accel_fname, 'r')
    in_vdot_str = in_vdot.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(3):
        self.q_v_vdot_robot[i][1][j + kVDotOffset] = float(in_vdot_str[str_index])
        str_index = str_index + 1
    in_vdot.close()

    #  Read in angular accelerations.
    in_alpha = open(angular_accel_fname, 'r')
    in_alpha_str = in_alpha.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(3):
        self.q_v_vdot_robot[i][1][j + kAlphaOffset] = float(in_alpha_str[str_index])
        str_index = str_index + 1
    '''

    #  Make sure there are no NaN's.
    for i in range(len(self.q_v_vdot_robot)):
      for j in range(len(self.q_v_vdot_robot[i][1])):
        assert not math.isnan(self.q_v_vdot_robot[i][1][j])


  def ReadContactPoint(self, timings_fname, cp_fname, cp_dot_fname):
    kLocationDim = 3
    kContactPointVelocityOffset = 3
    
    del self.contact_kinematics[:]

    #  Read in timings.
    in_timings = open(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
      t = float(in_timings_str[i])
      self.contact_kinematics.append(
          (t, np.ones((kLocationDim * 2)) * float('nan')))
    in_timings.close()

    #  Read in contact point location over time.
    in_x = open(cp_fname, 'r')
    in_x_str = in_x.read().split()
    str_index = 0
    for i in range(len(self.contact_kinematics)):
      for j in range(kLocationDim):
        self.contact_kinematics[i][1][j] = float(in_x_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.contact_kinematics) * kLocationDim
    in_x.close()
    
    #  Read in contact point velocity over time.
    in_xdot = open(cp_dot_fname, 'r')
    in_xdot_str = in_xdot.read().split()
    str_index = 0
    for i in range(len(self.contact_kinematics)):
      for j in range(kLocationDim):
        self.contact_kinematics[i][1][j + kContactPointVelocityOffset] =\
          float(in_xdot_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.contact_kinematics) * kLocationDim

  def str2bool(self, v):
    return v.lower() in ("yes", "true", "t", "1")

  def ReadBallQVAndVdot(self, timings_fname,
                        com_locations_fname,
                        quats_fname,
                        com_velocity_fname,
                        angular_velocity_fname,
                        com_accel_fname,
                        angular_accel_fname,
                        contact_indicator_fname):
    kStatePlusAccelDim = 19

    del self.q_v_vdot_ball[:]

    #  Read in timings.
    in_timings = open(timings_fname, 'r')
    in_timings_str = in_timings.read().split()
    for i in range(len(in_timings_str)):
        t = float(in_timings_str[i])
        self.q_v_vdot_ball.append((t, np.ones((kStatePlusAccelDim)) * float('nan')))
    in_timings.close()

    # Set offsets in the vector.
    kQuatOffset = 0
    kComOffset = 4
    kWOffset = 7
    kVOffset = 10
    kAlphaOffset = 13
    kVDotOffset = 16

    #  Read in com locations.
    in_x = open(com_locations_fname, 'r')
    in_x_str = in_x.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(3):
        self.q_v_vdot_ball[i][1][j + kComOffset] = float(in_x_str[str_index])
        str_index = str_index + 1
    assert str_index == len(self.q_v_vdot_ball)*3
    in_x.close()

    #  Read in unit quaternions.
    quat_tol = 1e-7
    in_quat = open(quats_fname, 'r')
    in_quat_str = in_quat.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(4):
        self.q_v_vdot_ball[i][1][j + kQuatOffset] = float(in_quat_str[str_index])
        str_index = str_index + 1
      qw = self.q_v_vdot_ball[i][1][kQuatOffset+0]
      qx = self.q_v_vdot_ball[i][1][kQuatOffset+1]
      qy = self.q_v_vdot_ball[i][1][kQuatOffset+2]
      qz = self.q_v_vdot_ball[i][1][kQuatOffset+3]
      assert abs(qw*qw + qx*qx + qy*qy + qz*qz - 1) < 1e-8
    assert str_index == len(self.q_v_vdot_ball)*4
    in_quat.close()

    #  Read in translational velocities.
    in_v = open(com_velocity_fname, 'r')
    in_v_str = in_v.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(3):
        self.q_v_vdot_ball[i][1][j + kVOffset] = float(in_v_str[str_index])
        str_index = str_index + 1
    in_v.close()

    #  Read in angular velocities.
    in_w = open(angular_velocity_fname, 'r')
    in_w_str = in_w.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(3):
        self.q_v_vdot_ball[i][1][j + kWOffset] = float(in_w_str[str_index])
        str_index = str_index + 1
    in_w.close()

    #  Read in translational acceleration.
    in_vdot = open(com_accel_fname, 'r')
    in_vdot_str = in_vdot.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(3):
        self.q_v_vdot_ball[i][1][j + kVDotOffset] = float(in_vdot_str[str_index])
        str_index = str_index + 1
    in_vdot.close()
    
    #  Read in angular accelerations.
    in_alpha = open(angular_accel_fname, 'r')
    in_alpha_str = in_alpha.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(3):
        self.q_v_vdot_ball[i][1][j + kAlphaOffset] = float(in_alpha_str[str_index])
        str_index = str_index + 1

    #  Make sure there are no NaN's.
    for i in range(len(self.q_v_vdot_ball)):
      for j in range(len(self.q_v_vdot_ball[i][1])):
        assert not math.isnan(self.q_v_vdot_ball[i][1][j])

    #  Read in the contact indicators.
    in_contact_indicator = open(contact_indicator_fname, 'r')
    in_contact_indicator_str = in_contact_indicator.read().split()
    str_index = 0
    for i in range(len(self.q_v_vdot_ball)):
      self.contact_desired.append((self.q_v_vdot_ball[i][0], self.str2bool(in_contact_indicator_str[str_index])))
      str_index = str_index + 1
    in_contact_indicator.close()

  def SearchBinary(self, t, vec):
    left = 0
    right = len(vec)-1
    while right - left > 1:
      mid = (left + right)/2
      if (t > vec[mid][0]):
        left = mid
      else:
        right = mid
    assert left != right
    return left
