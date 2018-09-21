#pragma once

#include <fstream>
#include <iostream>
#include <limits>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace examples {
namespace iiwa_soccer {

class ManipulationPlan {
 public:
  /// First seven variables of the vector correspond to robot configuration;
  /// the next seven variables of the vector correspond to joint velocity. The
  /// sole scalar value corresponds to the designated time.
  std::vector<std::pair<double, VectorX<double>>> q_qdot_qddot_robot_;

  /// First three variables: com position; next four variables: quaternion
  /// orientation (qw qx qy qz); next three variables: translational velocity
  /// (expressed in the world frame); next three variables: angular velocity
  /// (expressed in the world frame). The sole scalar value corresponds to the
  /// designated time.
  std::vector<std::pair<double, VectorX<double>>> q_v_vdot_ball_;

  /// Times and indicators of contact / no contact desired between the ball and
  /// the robot. `true` indicates contact desired.
  std::vector<std::pair<double, bool>> contact_desired_;

  /// The planned point of contact between the robot and the ball in the world
  /// frame (first three components), and the time derivative of that point
  /// (next three components). The sole scalar value corresponds to the
  /// designated time.
  std::vector<std::pair<double, VectorX<double>>> contact_kinematics_;

  /// Gets the final time of the plan.
  double get_end_time() const {
    double end_time = 0.0;
    if (!q_qdot_qddot_robot_.empty())
      end_time = std::max(end_time, q_qdot_qddot_robot_.back().first);
    if (!q_v_vdot_ball_.empty())
      end_time = std::max(end_time, q_v_vdot_ball_.back().first);
    if (!contact_desired_.empty())
      DRAKE_DEMAND(end_time >= contact_desired_.back().first);
    if (!contact_kinematics_.empty())
      DRAKE_DEMAND(end_time >= contact_kinematics_.back().first);
    return end_time;
  }

  /// Returns whether it is desired that the robot and ball be in contact at
  /// desired time (`t`).
  bool IsContactDesired(double t) const {
    return contact_desired_[SearchBinary(t, contact_desired_)].second;
  }

  /// Gets the ball q, v, and vdot at a particular point in time.
  VectorX<double> GetBallQVAndVdot(double t) const {
    return q_v_vdot_ball_[SearchBinary(t, q_v_vdot_ball_)].second;
  }

  /// Gets the ball q, v, and vdot at a particular point in time.
  VectorX<double> GetRobotQQdotAndQddot(double t) const {
    return q_qdot_qddot_robot_[SearchBinary(t, q_qdot_qddot_robot_)].second;
  }

  /// Gets the contact kinematics at a particular point in time.
  /// First three components are the contact point location in the global frame
  /// and the second three components are the time derivative of that location
  /// in the global frame.
  VectorX<double> GetContactKinematics(double t) const {
    return contact_kinematics_[SearchBinary(t, contact_kinematics_)].second;
  }

  void ReadRobotQQdotAndQddot(std::string timings_fname, std::string q_fname,
                              std::string qd_fname, std::string qdd_fname) {
    const int kDim = 21;

    q_qdot_qddot_robot_.clear();

    // Read in timings.
    std::ifstream in_timings(timings_fname.c_str());
    DRAKE_DEMAND(!in_timings.fail());
    while (true) {
      double t;
      in_timings >> t;
      if (in_timings.eof() || in_timings.fail())
        break;
      q_qdot_qddot_robot_.emplace_back(
          std::make_pair(t, VectorX<double>(kDim)));
      for (int i = 0; i < q_qdot_qddot_robot_.back().second.size(); ++i) {
        q_qdot_qddot_robot_.back().second[i] =
            std::numeric_limits<double>::quiet_NaN();
      }
    }
    in_timings.close();

    // Read in joint angles.
    std::ifstream in_q(q_fname.c_str());
    DRAKE_DEMAND(!in_q.fail());
    for (size_t i = 0; i < q_qdot_qddot_robot_.size(); ++i) {
      for (int j = 0; j < kDim/3; ++j) {
        in_q >> q_qdot_qddot_robot_[i].second[j];
        DRAKE_DEMAND(!in_q.eof() && !in_q.fail());
      }
    }

    // Make sure there is nothing else to read.
    double dummy;
    in_q >> dummy;
    DRAKE_DEMAND(in_q.eof() || in_q.fail());
    in_q.close();

    // Read in joint velocities.
    std::ifstream in_qd(qd_fname.c_str());
    DRAKE_DEMAND(!in_qd.fail());
    for (size_t i = 0; i < q_qdot_qddot_robot_.size(); ++i) {
      for (int j = 0; j < kDim/3; ++j) {
        in_qd >> q_qdot_qddot_robot_[i].second[j + kDim/3];
        DRAKE_DEMAND(!in_qd.eof() && !in_qd.fail());
      }
    }

    // Make sure there is nothing else to read.
    in_qd >> dummy;
    DRAKE_DEMAND(in_qd.eof() || in_qd.fail());
    in_qd.close();

    // Read in joint velocities.
    std::ifstream in_qdd(qdd_fname.c_str());
    DRAKE_DEMAND(!in_qdd.fail());
    for (size_t i = 0; i < q_qdot_qddot_robot_.size(); ++i) {
      for (int j = 0; j < kDim/3; ++j) {
        in_qdd >> q_qdot_qddot_robot_[i].second[j + 2 * kDim/3];
        DRAKE_DEMAND(!in_qdd.eof() && !in_qdd.fail());
      }
    }

    // Make sure there is nothing else to read.
    in_qdd >> dummy;
    DRAKE_DEMAND(in_qdd.eof() || in_qdd.fail());
    in_qdd.close();

    // Make sure there are no NaN's.
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i)
      for (int j = 0; j < q_v_vdot_ball_[i].second.size(); ++j)
        DRAKE_DEMAND(!std::isnan(q_qdot_qddot_robot_[i].second[j]));
  }

  void ReadContactPoint(std::string timings_fname, std::string cp_fname,
                        std::string cp_dot_fname) {
    const int kLocationDim = 3;
    const int kContactPointVelocityOffset = 3;
    
    contact_kinematics_.clear();

    // Read in timings.
    std::ifstream in_timings(timings_fname.c_str());
    DRAKE_DEMAND(!in_timings.fail());
    while (true) {
      double t;
      in_timings >> t;
      if (in_timings.eof() || in_timings.fail())
        break;
      contact_kinematics_.emplace_back(std::make_pair(t, VectorX<double>(
          kLocationDim * 2)));
      for (int i = 0; i < contact_kinematics_.back().second.size(); ++i) {
        contact_kinematics_.back().second[i] =
            std::numeric_limits<double>::quiet_NaN();
      }
    }
    in_timings.close();

    // Read in contact point location over time.
    std::ifstream in_x(cp_fname.c_str());
    DRAKE_DEMAND(!in_x.fail());
    for (size_t i = 0; i < contact_kinematics_.size(); ++i) {
      for (int j = 0; j < kLocationDim; ++j) {
        in_x >> contact_kinematics_[i].second[j];
        DRAKE_DEMAND(!in_x.eof() && !in_x.fail());
      }
    }

    // Make sure there is nothing else to read.
    double dummy;
    in_x >> dummy;
    DRAKE_DEMAND(in_x.eof() || in_x.fail());
    in_x.close();
    
    // Read in contact point velocity over time.
    std::ifstream in_xdot(cp_dot_fname.c_str());
    DRAKE_DEMAND(!in_xdot.fail());
    for (size_t i = 0; i < contact_kinematics_.size(); ++i) {
      for (int j = 0; j < kLocationDim; ++j) {
        in_xdot >> contact_kinematics_[i].second[j + kContactPointVelocityOffset];
        DRAKE_DEMAND(!in_xdot.eof() && !in_xdot.fail());
      }
    }

    // Make sure there is nothing else to read.
    in_xdot >> dummy;
    DRAKE_DEMAND(in_xdot.eof() || in_xdot.fail());
    in_xdot.close();
  }

  void ReadBallQVAndVdot(std::string timings_fname,
                         std::string com_locations_fname,
                         std::string quats_fname,
                         std::string com_velocity_fname,
                         std::string angular_velocity_fname,
                         std::string com_accel_fname,
                         std::string angular_accel_fname,
                         std::string contact_indicator_fname) {
    const int kStatePlusAccelDim = 19;

    q_v_vdot_ball_.clear();

    // Read in timings.
    std::ifstream in_timings(timings_fname.c_str());
    DRAKE_DEMAND(!in_timings.fail());
    while (true) {
      double t;
      in_timings >> t;
      if (in_timings.eof() || in_timings.fail())
        break;
      q_v_vdot_ball_.emplace_back(
          std::make_pair(t, VectorX<double>(kStatePlusAccelDim)));
      for (int i = 0; i < q_v_vdot_ball_.back().second.size(); ++i) {
        q_v_vdot_ball_.back().second[i] =
            std::numeric_limits<double>::quiet_NaN();
      }
    }
    in_timings.close();

    // Read in com locations.
    std::ifstream in_x(com_locations_fname.c_str());
    DRAKE_DEMAND(!in_x.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_x >> q_v_vdot_ball_[i].second[j];
        DRAKE_DEMAND(!in_x.eof() && !in_x.fail());
      }
    }

    // Make sure there is nothing else to read.
    double dummy;
    in_x >> dummy;
    DRAKE_DEMAND(in_x.eof() || in_x.fail());
    in_x.close();

    // Read in unit quaternions.
    const double quat_tol = 1e-7;
    const int kQuatOffset = 3;
    std::ifstream in_quat(quats_fname.c_str());
    DRAKE_DEMAND(!in_quat.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      for (int j = 0; j < 4; ++j) {
        in_quat >> q_v_vdot_ball_[i].second[j + kQuatOffset];
        DRAKE_DEMAND(!in_quat.eof() && !in_quat.fail());
      }
      DRAKE_DEMAND(std::abs(q_v_vdot_ball_[i].second.segment(
          kQuatOffset, 4).norm() - 1.0) < quat_tol);
    }

    // Make sure there is nothing else to read.
    in_quat >> dummy;
    DRAKE_DEMAND(in_quat.eof() || in_quat.fail());
    in_quat.close();

    // Read in translational velocities.
    const int kVOffset = 7;
    std::ifstream in_v(com_velocity_fname.c_str());
    DRAKE_DEMAND(!in_v.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_v >> q_v_vdot_ball_[i].second[j + kVOffset];
        DRAKE_DEMAND(!in_v.eof() && !in_v.fail());
      }
    }

    // Make sure there is nothing else to read.
    in_v >> dummy;
    DRAKE_DEMAND(in_v.eof() || in_v.fail());
    in_v.close();

    // Read in angular velocities.
    const int kWOffset = 10;
    std::ifstream in_w(angular_velocity_fname.c_str());
    DRAKE_DEMAND(!in_w.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_w >> q_v_vdot_ball_[i].second[j + kWOffset];
        DRAKE_DEMAND(!in_w.eof() && !in_w.fail());
      }
    }

    // Make sure there is nothing else to read.
    in_w >> dummy;
    DRAKE_DEMAND(in_w.eof() || in_w.fail());
    in_w.close();

    // Read in translational acceleration.
    const int kVDotOffset = 13;
    std::ifstream in_vdot(com_accel_fname.c_str());
    DRAKE_DEMAND(!in_vdot.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_vdot >> q_v_vdot_ball_[i].second[j + kVDotOffset];
        DRAKE_DEMAND(!in_vdot.eof() && !in_vdot.fail());
      }
    }

    // Make sure there is nothing else to read.
    in_vdot >> dummy;
    DRAKE_DEMAND(in_vdot.eof() || in_vdot.fail());
    in_vdot.close();
    
    // Read in angular accelerations.
    const int kAlphaOffset = 16;
    std::ifstream in_alpha(angular_accel_fname.c_str());
    DRAKE_DEMAND(!in_alpha.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        in_alpha >> q_v_vdot_ball_[i].second[j + kAlphaOffset];
        DRAKE_DEMAND(!in_alpha.eof() && !in_alpha.fail());
      }
    }

    // Make sure there are no NaN's.
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i)
      for (int j = 0; j < q_v_vdot_ball_[i].second.size(); ++j)
        DRAKE_DEMAND(!std::isnan(q_v_vdot_ball_[i].second[j]));

    // Make sure there is nothing else to read.
    in_alpha >> dummy;
    DRAKE_DEMAND(in_alpha.eof() || in_alpha.fail());
    in_alpha.close();
    
    // Read in the contact indicators.
    std::ifstream in_contact_indicator(contact_indicator_fname.c_str());
    DRAKE_DEMAND(!in_contact_indicator.fail());
    for (size_t i = 0; i < q_v_vdot_ball_.size(); ++i) {
      bool status;
      in_contact_indicator >> status;
      DRAKE_DEMAND(!in_contact_indicator.eof() && !in_contact_indicator.fail());
      contact_desired_.emplace_back(
          std::make_pair(q_v_vdot_ball_[i].first, status));
    }
    in_contact_indicator.close();
  }

 private:
  // Performs a binary search on the desired array for the given time.
  template <class T>
  int SearchBinary(
      double t, const std::vector<std::pair<double, T>>& vec) const {
    int left = 0, right = static_cast<int>(vec.size()-1);
    while (right - left > 1) {
      int mid = (left + right)/2;
      if (t > vec[mid].first) {
        left = mid;
      } else {
        right = mid;
      }
    }
    DRAKE_DEMAND(left != right);
    return left;
  }
};

}  // iiwa_soccer
}  // examples
}  // drake
