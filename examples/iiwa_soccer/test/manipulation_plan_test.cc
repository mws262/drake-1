#include "drake/examples/iiwa_soccer/manipulation_plan.h"

#include <cmath>

#include <gtest/gtest.h>

namespace drake {
namespace examples {
namespace iiwa_soccer {
namespace {

GTEST_TEST(ManipulationPlan, RobotStatesRead) {
  ManipulationPlan plan;
  plan.ReadRobotQQdotAndQddot(
      "examples/iiwa_soccer/plan/joint_fit_timings.mat",
      "examples/iiwa_soccer/plan/joint_angle_fit.mat",
      "examples/iiwa_soccer/plan/joint_vel_fit.mat",
      "examples/iiwa_soccer/plan/joint_accel_fit.mat");
}

GTEST_TEST(ManipulationPlan, ContactPointsRead) {
  ManipulationPlan plan;
  plan.ReadContactPoint("examples/iiwa_soccer/plan/contact_pt_timings.mat",
                        "examples/iiwa_soccer/plan/contact_pt_positions.mat",
                        "examples/iiwa_soccer/plan/contact_pt_velocities.mat");
}

GTEST_TEST(ManipulationPlan, BallStatesRead) {
  ManipulationPlan plan;
  plan.ReadBallQVAndVdot(
      "examples/iiwa_soccer/plan/ball_timings.mat",
      "examples/iiwa_soccer/plan/ball_com_positions.mat",
      "examples/iiwa_soccer/plan/ball_quats.mat",
      "examples/iiwa_soccer/plan/ball_com_velocities.mat",
      "examples/iiwa_soccer/plan/ball_omegas.mat",
      "examples/iiwa_soccer/plan/ball_com_accelerations.mat",
      "examples/iiwa_soccer/plan/ball_alphas.mat",
      "examples/iiwa_soccer/plan/contact_status.mat");
}

}  // namespace
}  // namespace iiwa_soccer
}  // namespace examples
}  // namespace drake
