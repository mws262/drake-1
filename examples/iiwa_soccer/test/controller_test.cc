#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/iiwa_soccer/controller.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree_construction.h"

#include <cmath>

#include <gtest/gtest.h>

using Eigen::Matrix3Xd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using drake::parsers::urdf::AddModelInstanceFromUrdfFileToWorld;
using drake::systems::SystemOutput;
using drake::systems::RigidBodyPlant;

namespace drake {
namespace examples {
namespace iiwa_soccer {

// Debugging function for computing the translational velocity of a shared
// contact point (on the robot and soccer ball).
Vector3d CalcVelocity(
    const drake::multibody::collision::PointPair<double>& contact,
    const RigidBodyTree<double>& robot_and_ball_tree,
    const VectorXd& v,
    const KinematicsCache<double>& kinematics_cache) {
  // Get the two body indices.
  const int body_a_index = contact.elementA->get_body()->get_body_index();
  const int body_b_index = contact.elementB->get_body()->get_body_index();

  // The reported point on A's surface (As) in the world frame (W).
  const Vector3d p_WAs =
      kinematics_cache.get_element(body_a_index).transform_to_world *
          contact.ptA;

  // The reported point on B's surface (Bs) in the world frame (W).
  const Vector3d p_WBs =
      kinematics_cache.get_element(body_b_index).transform_to_world *
          contact.ptB;

  // Get the point of contact in the world frame.
  const Vector3d p_W = (p_WAs + p_WBs) * 0.5;

  // The contact point in A's frame.
  const auto X_AW = kinematics_cache.get_element(body_a_index)
      .transform_to_world.inverse(Eigen::Isometry);
  const Vector3d p_A = X_AW * p_W;

  // The contact point in B's frame.
  const auto X_BW = kinematics_cache.get_element(body_b_index)
      .transform_to_world.inverse(Eigen::Isometry);
  const Vector3d p_B = X_BW * p_W;

  // Get the Jacobian matrices.
  const auto JA = robot_and_ball_tree.transformPointsJacobian(
      kinematics_cache, p_A, body_a_index, 0, false);
  const auto JB = robot_and_ball_tree.transformPointsJacobian(
      kinematics_cache, p_B, body_b_index, 0, false);

  // Compute the translational velocity.
  return (JA * v) - (JB * v);
}

// Debugging function for computing the translational velocity of a shared
// contact point (on the robot and soccer ball).
Vector3d CalcAcceleration(
    const drake::multibody::collision::PointPair<double>& contact,
    const RigidBodyTree<double>& robot_and_ball_tree,
    const VectorXd& v,
    const VectorXd& vdot,
    const KinematicsCache<double>& kinematics_cache) {
  // Get the two body indices.
  const int body_a_index = contact.elementA->get_body()->get_body_index();
  const int body_b_index = contact.elementB->get_body()->get_body_index();

  // The reported point on A's surface (As) in the world frame (W).
  const Vector3d p_WAs =
      kinematics_cache.get_element(body_a_index).transform_to_world *
          contact.ptA;

  // The reported point on B's surface (Bs) in the world frame (W).
  const Vector3d p_WBs =
      kinematics_cache.get_element(body_b_index).transform_to_world *
          contact.ptB;

  // Get the point of contact in the world frame.
  const Vector3d p_W = (p_WAs + p_WBs) * 0.5;

  // The contact point in A's frame.
  const auto X_AW = kinematics_cache.get_element(body_a_index)
      .transform_to_world.inverse(Eigen::Isometry);
  const Vector3d p_A = X_AW * p_W;

  // The contact point in B's frame.
  const auto X_BW = kinematics_cache.get_element(body_b_index)
      .transform_to_world.inverse(Eigen::Isometry);
  const Vector3d p_B = X_BW * p_W;

  // Get the Jacobian matrices.
  const auto JA = robot_and_ball_tree.transformPointsJacobian(
      kinematics_cache, p_A, body_a_index, 0, false);
  const auto JB = robot_and_ball_tree.transformPointsJacobian(
      kinematics_cache, p_B, body_b_index, 0, false);

  // Get the Jacobian matrices times v.
  const auto JA_dot_v = robot_and_ball_tree.transformPointsJacobianDotTimesV(
      kinematics_cache, p_A, body_a_index, 0);
  const auto JB_dot_v = robot_and_ball_tree.transformPointsJacobianDotTimesV(
      kinematics_cache, p_B, body_b_index, 0);

  // Compute the acceleration.
  return (JA * vdot) - (JB * vdot) + (JA_dot_v - JB_dot_v);
}

// Debugging function for checking torque computation.
VectorX<double> ComputeTorque(const RigidBodyTree<double>& tree,
                              const VectorX<double>& q,
                              const VectorX<double>& v,
                              const VectorX<double>& vd_d) {
  // Compute the expected torque.
  KinematicsCache<double> cache = tree.doKinematics(q, v);
  eigen_aligned_std_unordered_map<RigidBody<double> const*,
                                  drake::TwistVector<double>> f_ext;

  return tree.massMatrix(cache) * vd_d + tree.dynamicsBiasTerm(cache, f_ext);
}

class ControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* armModelPath =
        "drake/examples/iiwa_soccer/models/iiwa14_spheres_collision.urdf";
    const char* ballModelPath =
        "drake/examples/iiwa_soccer/models/soccer_ball.urdf";

    // Create and import bodies into the trees.
    robot_and_ball_tree_ = std::make_unique<RigidBodyTree<double>>();
    robot_tree_ = std::make_unique<RigidBodyTree<double>>();

    // Import arm to trees.
    auto kuka_id_table =
        AddModelInstanceFromUrdfFileToWorld(FindResourceOrThrow(armModelPath),
                                            multibody::joints::kFixed,
                                            robot_and_ball_tree_.get());
    AddModelInstanceFromUrdfFileToWorld(FindResourceOrThrow(armModelPath),
                                        multibody::joints::kFixed,
                                        robot_tree_.get());

    // Import ball to tree.
    auto ball_id_table =
        AddModelInstanceFromUrdfFileToWorld(FindResourceOrThrow(ballModelPath),
                                            multibody::joints::kQuaternion,
                                            robot_and_ball_tree_.get());

    // Make ground part of tree.
    multibody::AddFlatTerrainToWorld(robot_and_ball_tree_.get(), 100., 10.);

    // Compile the two trees.
    robot_and_ball_tree_->compile();
    robot_tree_->compile();

    // Cartesian gains.
    Vector3<double> k_p = Vector3<double>(100.0, 100.0, 100.0);
    Vector3<double> k_d = Vector3<double>(10.0, 10.0, 10.0);

    // Sets pid gains.
    const int dim = 7;  // Number of actuators.
    joint_kp_.resize(dim);
    joint_ki_.resize(dim);
    joint_kp_ << 1, 2, 3, 4, 5, 6, 7;
    joint_ki_ << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7;
    joint_kd_ = joint_kp_ / 2.;

    // Create the controller.
    controller_ = std::make_unique<Controller>(
        *robot_and_ball_tree_.get(), *robot_tree_.get(),
        k_p, k_d, joint_kp_, joint_ki_, joint_kd_);

    // Create the plant and the context.
    context_ = controller_->CreateDefaultContext();
    output_ = controller_->AllocateOutput(*context_);
  }

  // Sets the states for the robot and the ball to those planned.
  void SetStates(double t, VectorXd* q, VectorXd* v) {
    const ManipulationPlan& plan = controller_->plan();
    q->resize(nq_ball_ + nq_robot_);
    v->resize(nv_ball_ + nqd_robot_);

    // Get the planned q, qdot, and v.
    const VectorXd q_robot_des = plan.GetRobotQQdotAndQddot(t).head(nq_robot_);
    const VectorXd qd_robot_des =
            plan.GetRobotQQdotAndQddot(t).segment(nq_robot_, nqd_robot_);
    const VectorXd q_ball_des = plan.GetBallQVAndVdot(t).head(nq_ball_);
    const VectorXd v_ball_des = plan.GetBallQVAndVdot(t).segment(
        nq_ball_, nv_ball_);

    // Set q and v.
    q->segment(controller_->get_robot_position_start_index_in_q(), nq_robot_) =
        q_robot_des;
    q->segment(controller_->get_ball_position_start_index_in_q(), nq_ball_) =
        q_ball_des;
    v->segment(controller_->get_robot_velocity_start_index_in_v(),
        nqd_robot_) = qd_robot_des;
    v->segment(controller_->get_ball_velocity_start_index_in_v(), nv_ball_).
        tail(3) = v_ball_des.head(3);
    v->segment(controller_->get_ball_velocity_start_index_in_v(), nv_ball_).
        head(3) = v_ball_des.tail(3);

    // Construct the robot reference positions and velocities and set them equal
    // to the current positions and velocities.
    // Get the robot reference positions, velocities, and accelerations.
    auto robot_q_input = std::make_unique<BasicVector<double>>(nq_robot_);
    auto robot_qd_input = std::make_unique<BasicVector<double>>(nqd_robot_);
    robot_q_input->get_mutable_value() << q_robot_des;
    robot_qd_input->get_mutable_value() << qd_robot_des;

    // Get the ball reference positions, velocities, and accelerations, and
    // set them equal to the estimated positions and velocities.
    auto ball_q_input = std::make_unique<BasicVector<double>>(nq_ball_);
    auto ball_v_input = std::make_unique<BasicVector<double>>(nv_ball_);
    ball_q_input->get_mutable_value() << q_ball_des;
    ball_v_input->get_mutable_value() << v_ball_des;

    // Sanity check.
    for (int i = 0; i < q_robot_des.size(); ++i) {
      EXPECT_FALSE(std::isnan(q_robot_des[i]));
      EXPECT_FALSE(std::isnan(qd_robot_des[i]));
    }

    // Set the robot and ball positions in velocities in the inputs.
    context_->FixInputPort(
        controller_->get_input_port_estimated_robot_q().get_index(),
        std::move(robot_q_input));
    context_->FixInputPort(
        controller_->get_input_port_estimated_robot_qd().get_index(),
        std::move(robot_qd_input));
    context_->FixInputPort(
        controller_->get_input_port_estimated_ball_q().get_index(),
        std::move(ball_q_input));
    context_->FixInputPort(
        controller_->get_input_port_estimated_ball_v().get_index(),
        std::move(ball_v_input));

    // Update the context.
    context_->set_time(t);
  }

  VectorXd joint_kp_, joint_ki_, joint_kd_;
  std::unique_ptr<Controller> controller_;
  std::unique_ptr<RigidBodyTree<double>> robot_tree_;
  std::unique_ptr<RigidBodyTree<double>> robot_and_ball_tree_;
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<SystemOutput<double>> output_;
  const int nq_robot_ = 7, nqd_robot_ = 7, nq_ball_ = 7, nv_ball_ = 6;
};

// Check that control without contact is correct.
// Tests the computed torque from InverseDynamicsController matches hand
// derived results for the kuka iiwa arm at a given state (q, v), when
// asked to track reference state (q_r, v_r) and reference acceleration (vd_r).
TEST_F(ControllerTest, TestTorque) {
  // Get the plan.
  const ManipulationPlan& plan = controller_->plan();

  // Advance time, finding a point at which contact is not desired.
  const double dt = 1e-3;
  double t = 0.0;
  const double t_final = plan.get_end_time();
  while (true) {
    if (!plan.IsContactDesired(t))
      break;
    t += dt;
    ASSERT_LT(t, t_final);
  }

  // Construct the robot positions and velocities, and inputs.
  VectorX<double> q(nq_robot_), qd(nqd_robot_);
  auto robot_q_input = std::make_unique<BasicVector<double>>(nq_robot_);
  auto robot_qd_input = std::make_unique<BasicVector<double>>(nqd_robot_);
  q << 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3;
  qd = q * 3;
  robot_q_input->get_mutable_value() << q;
  robot_qd_input->get_mutable_value() << qd;

  // Get the robot reference positions, velocities, and accelerations.
  const VectorXd q_robot_des = plan.GetRobotQQdotAndQddot(t).head(nq_robot_);
  const VectorXd qd_robot_des = plan.GetRobotQQdotAndQddot(t).
      segment(nq_robot_, nqd_robot_);
  const VectorXd qdd_robot_des = plan.GetRobotQQdotAndQddot(t).tail(nqd_robot_);

  // Sanity check.
  for (int i = 0; i < q_robot_des.size(); ++i) {
    EXPECT_FALSE(std::isnan(q_robot_des[i]));
    EXPECT_FALSE(std::isnan(qd_robot_des[i]));
    EXPECT_FALSE(std::isnan(qdd_robot_des[i]));
  }

  // Set dummy inputs for the ball.
  auto ball_q_input = std::make_unique<BasicVector<double>>(nq_ball_);
  ball_q_input->get_mutable_value() << VectorXd::Zero(nq_ball_);
  auto ball_v_input = std::make_unique<BasicVector<double>>(nv_ball_);
  ball_v_input->get_mutable_value() << VectorXd::Zero(nv_ball_);

  // Set the robot and ball positions in velocities in the inputs.
  context_->FixInputPort(
      controller_->get_input_port_estimated_robot_q().get_index(),
      std::move(robot_q_input));
  context_->FixInputPort(
      controller_->get_input_port_estimated_robot_qd().get_index(),
      std::move(robot_qd_input));
  context_->FixInputPort(
      controller_->get_input_port_estimated_ball_q().get_index(),
      std::move(ball_q_input));
  context_->FixInputPort(
      controller_->get_input_port_estimated_ball_v().get_index(),
      std::move(ball_v_input));

  // Set the integrated position error.
  const int num_actuators = 7;
  VectorX<double> q_int(num_actuators);
  q_int << -1, -2, -3, -4, -5, -6, -7;
  controller_->set_integral_value(context_.get(), q_int);

  // Compute the output from the controller.
  controller_->CalcOutput(*context_, output_.get());

  // The results should equal to this.
  VectorX<double> vd_d =
      (joint_kp_.array() * (q_robot_des - q).array()).matrix() +
      (joint_kd_.array() * (qd_robot_des - qd).array()).matrix() +
      (joint_ki_.array() * q_int.array()).matrix() + qdd_robot_des;

  VectorX<double> expected_torque = ComputeTorque(
      controller_->get_robot_tree(), q, qd, vd_d);

  // Checks the expected and computed gravity torque.
  const BasicVector<double>* output_vector = output_->get_vector_data(0);
  EXPECT_TRUE(CompareMatrices(expected_torque, output_vector->get_value(),
                              1e-10, MatrixCompareType::absolute));
}

// Check control outputs for when robot is not in contact with the ball but it
// is desired to be.
TEST_F(ControllerTest, NoContactButContactIntendedOutputsCorrect) {
  // Get the plan.
  const ManipulationPlan& plan = controller_->plan();

  // Advance time, finding a point at which contact is desired *and* where there
  // is at least a single contact.
  const double dt = 1e-3;
  double t = 0.0;
  const double t_final = plan.get_end_time();
  VectorXd q(nq_ball_ + nq_robot_), v(nv_ball_ + nqd_robot_);
  while (true) {
    if (plan.IsContactDesired(t)) {
      SetStates(t, &q, &v);

      // Construct the kinematics cache.
      auto kinematics_cache =
          controller_->get_robot_and_ball_tree().doKinematics(q, v);

      // Look for contact.
      std::vector<drake::multibody::collision::PointPair<double>> contacts =
          const_cast<RigidBodyTree<double>*>(
              &controller_->get_robot_and_ball_tree())
              ->ComputeMaximumDepthCollisionPoints(kinematics_cache, true);
      if (contacts.empty())
        break;
    }

    // No contact desired or contact was found.
    t += dt;
    ASSERT_LT(t, t_final);
  }

  // Compute the output from the controller.
  controller_->CalcOutput(*context_, output_.get());
}

// Check that Jacobian construction is correct.
TEST_F(ControllerTest, JacobianConstruction) {
  // Get the plan.
  const ManipulationPlan& plan = controller_->plan();

  // Advance time, finding a point at which contact is desired *and* where there
  // is at least a single contact.
  const double dt = 1e-3;
  double t = 0.0;
  const double t_final = plan.get_end_time();
  VectorXd q(nq_ball_ + nq_robot_), v(nv_ball_ + nqd_robot_);
  while (true) {
    if (plan.IsContactDesired(t)) {
      SetStates(t, &q, &v);

      // Construct the kinematics cache.
      auto kinematics_cache =
          controller_->get_robot_and_ball_tree().doKinematics(q, v);

      // Look for contact.
      std::vector<drake::multibody::collision::PointPair<double>> contacts =
          const_cast<RigidBodyTree<double>*>(
              &controller_->get_robot_and_ball_tree())
              ->ComputeMaximumDepthCollisionPoints(kinematics_cache, true);
      if (contacts.size() == 1)
        break;
    }

    // No contact desired or exactly one contact not found.
    t += dt;
    ASSERT_LT(t, t_final);
  }

  // Recompute the kinematics cache and determine the contact again.
  auto kinematics_cache =
      controller_->get_robot_and_ball_tree().doKinematics(q, v);
  std::vector<drake::multibody::collision::PointPair<double>> contacts =
      const_cast<RigidBodyTree<double>*>(
          &controller_->get_robot_and_ball_tree())
          ->ComputeMaximumDepthCollisionPoints(kinematics_cache, true);
  ASSERT_EQ(contacts.size(), 1);

  // Construct the Jacobian matrices using the controller function.
  MatrixXd N, S, T, Ndot, Sdot, Tdot;
  controller_->ConstructJacobians(contacts, kinematics_cache,
                                  &N, &S, &T, &Ndot, &Sdot, &Tdot);
  EXPECT_EQ(Ndot.rows(), contacts.size());
  EXPECT_EQ(Ndot.cols(), contacts.size());

  // Compute the velocity using the Jacobian matrices.
  const VectorXd Nv = N * v;
  ASSERT_EQ(contacts.size(), Nv.size());

  // Check the velocity against the direction specified by the contact normal.
  const Vector3d pv = CalcVelocity(
      contacts.front(), controller_->get_robot_and_ball_tree(), v,
      kinematics_cache);

  // Set an arbitrary acceleration.
  VectorXd vdot(v.size());
  for (int i = 0; i < vdot.size(); ++i)
    vdot[i] = i + 1;

  // Check the acceleration against the direction specified by the contact
  // normal.
  const Vector3d pvdot = CalcAcceleration(
      contacts.front(), controller_->get_robot_and_ball_tree(), v, vdot,
      kinematics_cache);

  // Compare the velocities and accelerations.
  EXPECT_NEAR(pv.dot(contacts.front().normal), Nv[0], 1e-14);
  EXPECT_NEAR(pvdot.dot(contacts.front().normal),
              (N * vdot)[0] + Ndot(0,0), 1e-14);

  // Set the parts of `v` corresponding to the robot to zero, so as not to
  // affect the velocity calculation.
  v.segment(controller_->get_robot_velocity_start_index_in_v(),
            nqd_robot_).setZero();

  // Kinematic cache must be recomputed.
  kinematics_cache =
      controller_->get_robot_and_ball_tree().doKinematics(q, v);

  // Set the contact point to an arbitrary point in the world.
  const Vector3d p(1, 2, 3);

  // Set the contact point in the body frames.
  const int body_a_index = contacts.front().
      elementA->get_body()->get_body_index();
  const int body_b_index = contacts.front().
      elementB->get_body()->get_body_index();
  contacts.front().ptA = kinematics_cache.get_element(body_a_index).
      transform_to_world.inverse(Eigen::Isometry) * p;
  contacts.front().ptB = kinematics_cache.get_element(body_b_index).
      transform_to_world.inverse(Eigen::Isometry) * p;

  // Get the body-to-world transformation for the ball.
  const auto& wTb = kinematics_cache.
      get_element(controller_->get_robot_and_ball_tree().
      FindBodyIndex("ball")).transform_to_world;

  // We know that the velocity of a point on the ball is v + ω × (p - x),
  // where x is the center of mass of the ball, v = dx/dt, ω is the angular
  // velocity, and p is the point on the ball. All quantities are expressed
  // in the global frame.
  const Vector3d x = q.segment(
      controller_->get_ball_position_start_index_in_q(), nq_ball_).head(3);
  const Vector3d omega = wTb.linear() * v.segment(
      controller_->get_ball_velocity_start_index_in_v(), nv_ball_).head(3);
  const Vector3d xd = wTb.linear() * v.segment(
      controller_->get_ball_velocity_start_index_in_v(), nv_ball_).tail(3);

  // Compute the velocity.
  const Vector3d pv_test = CalcVelocity(
      contacts.front(), controller_->get_robot_and_ball_tree(), v,
      kinematics_cache);

  // Check the velocity.
  EXPECT_LT((pv_test - (xd + omega.cross(p - x))).norm(), 1e-12);
}

// Check that velocity at the contact point remains sufficiently close to zero.
TEST_F(ControllerTest, ZeroVelocityAtContact) {
  // Get the plan.
  const ManipulationPlan& plan = controller_->plan();

  // Advance time, finding a point at which contact is desired *and* where there
  // is at least a single contact.
  const double dt = 1e-3;
  double t = 0.0;
  const double t_final = plan.get_end_time();
  VectorXd q(nq_ball_ + nq_robot_), v(nv_ball_ + nqd_robot_);
  while (true) {
    // Set the planned q and v.
    if (plan.IsContactDesired(t)) {
      SetStates(t, &q, &v);

      // Construct the kinematics cache.
      auto kinematics_cache =
          controller_->get_robot_and_ball_tree().doKinematics(q, v);

      // Look for contact.
      std::vector<drake::multibody::collision::PointPair<double>> contacts =
          const_cast<RigidBodyTree<double>*>(
              &controller_->get_robot_and_ball_tree())
              ->ComputeMaximumDepthCollisionPoints(kinematics_cache, true);
      if (contacts.size() == 2)
        break;
    }

    // No contact desired or exactly one contact not found.
    t += dt;
    ASSERT_LT(t, t_final);
  }

  // Recompute the kinematics cache and determine the contact again.
  auto kinematics_cache =
      controller_->get_robot_and_ball_tree().doKinematics(q, v);
  std::vector<drake::multibody::collision::PointPair<double>> contacts =
      const_cast<RigidBodyTree<double>*>(
          &controller_->get_robot_and_ball_tree())
          ->ComputeMaximumDepthCollisionPoints(kinematics_cache, true);

  // Get the *planned* point of contact in the world.
  // Set the contact point to an arbitrary point in the world.
  const Vector3d p = plan.GetContactKinematics(t).head(3);

  // Set the contact point in the body frames.
  const int body_a_index = contacts.front().
      elementA->get_body()->get_body_index();
  const int body_b_index = contacts.front().
      elementB->get_body()->get_body_index();
  contacts.front().ptA = kinematics_cache.get_element(body_a_index).
      transform_to_world.inverse(Eigen::Isometry) * p;
  contacts.front().ptB = kinematics_cache.get_element(body_b_index).
      transform_to_world.inverse(Eigen::Isometry) * p;

  // Construct the Jacobian matrices using the controller function.
  MatrixXd N, S, T, Ndot, Sdot, Tdot;
  controller_->ConstructJacobians(contacts, kinematics_cache,
                                  &N, &S, &T, &Ndot, &Sdot, &Tdot);
  EXPECT_EQ(Ndot.rows(), contacts.size());
  EXPECT_EQ(Ndot.cols(), 1);

  // Verify that the velocity at the contact point is approximately zero.
  const double zero_velocity_tol = 1e-12;
  const VectorXd Nv = N * v;
  const VectorXd Sv = S * v;
  const VectorXd Tv = T * v;
  ASSERT_EQ(contacts.size(), Nv.size());
  ASSERT_EQ(contacts.size(), Sv.size());
  ASSERT_EQ(contacts.size(), Tv.size());
  EXPECT_LT(Nv.norm(), zero_velocity_tol);
  EXPECT_LT(Sv.norm(), zero_velocity_tol);
  EXPECT_LT(Tv.norm(), zero_velocity_tol);
}

// Check that the contact distance when the plan indicates contact is desired
// always lies below a threshold.
TEST_F(ControllerTest, ContactDistanceBelowThreshold) {
  // Get the plan.
  const ManipulationPlan& plan = controller_->plan();

  // Advance time, finding a point at which contact is desired.
  const double dt = 1e-3;
  double t = 0.0;
  VectorXd q(nq_ball_ + nq_robot_), v(nv_ball_ + nqd_robot_);
  while (true) {
    // Determine whether the plan indicates contact is desired.
    if (!plan.IsContactDesired(t)) {
      t += dt;
      continue;
    }

    // Set the states to q and v.
    SetStates(t, &q, &v);

    // Get the robot and ball tree as non-const, which is necessary to call some
    // collision detection routines.
    RigidBodyTree<double>* robot_and_ball_tree_nc =
        const_cast<RigidBodyTree<double>*>(
            &controller_->get_robot_and_ball_tree());

    // Compute the kinematics cache.
    auto kinematics_cache =
        controller_->get_robot_and_ball_tree().doKinematics(q, v);

    // Get the collision IDs for the robot foot and the ball.
    std::vector<drake::multibody::collision::ElementId> ids_to_check;
    controller_->get_ball_from_robot_and_ball_tree().
        appendCollisionElementIdsFromThisBody(ids_to_check);
    const auto& robot_link6 = controller_->
        get_robot_link6_from_robot_and_ball_tree();
    robot_link6.appendCollisionElementIdsFromThisBody(ids_to_check);

    // Get the closest points on the robot foot and the ball.
    kinematics_cache = robot_and_ball_tree_nc->doKinematics(q, v);
    std::vector<int> indices_A, indices_B;
    VectorXd phi;
    Matrix3Xd points_A, points_B, normal;
    bool result = robot_and_ball_tree_nc->collisionDetect(
        kinematics_cache, phi, normal, points_A, points_B, indices_A,
        indices_B, ids_to_check, true);
    DRAKE_DEMAND(result);
    DRAKE_DEMAND(phi.size() == 1);
    DRAKE_DEMAND(indices_A.size() == 1);
    DRAKE_DEMAND(indices_B.size() == 1);
    const double dist_link6 = phi[0];

    // Redetermine the collision IDs.
    ids_to_check.clear();
    controller_->get_ball_from_robot_and_ball_tree().
        appendCollisionElementIdsFromThisBody(ids_to_check);
    const auto& robot_link7 = controller_->
        get_robot_link7_from_robot_and_ball_tree();
    robot_link7.appendCollisionElementIdsFromThisBody(ids_to_check);

    // Get the closest points on the robot foot and the ball.
    kinematics_cache = robot_and_ball_tree_nc->doKinematics(q, v);
    indices_A.clear();
    indices_B.clear();
    result = robot_and_ball_tree_nc->collisionDetect(
        kinematics_cache, phi, normal, points_A, points_B, indices_A,
        indices_B, ids_to_check, true);
    DRAKE_DEMAND(result);
    DRAKE_DEMAND(phi.size() == 1);
    DRAKE_DEMAND(indices_A.size() == 1);
    DRAKE_DEMAND(indices_B.size() == 1);
    const double dist_link7 = phi[0];

    // Redetermine the collision IDs.
    ids_to_check.clear();
    controller_->get_ball_from_robot_and_ball_tree().
        appendCollisionElementIdsFromThisBody(ids_to_check);
    const auto& robot_link5 = controller_->
        get_robot_link5_from_robot_and_ball_tree();
    robot_link5.appendCollisionElementIdsFromThisBody(ids_to_check);

    // Get the closest points on the robot foot and the ball.
    kinematics_cache = robot_and_ball_tree_nc->doKinematics(q, v);
    indices_A.clear();
    indices_B.clear();
    result = robot_and_ball_tree_nc->collisionDetect(
        kinematics_cache, phi, normal, points_A, points_B, indices_A,
        indices_B, ids_to_check, true);
    DRAKE_DEMAND(result);
    DRAKE_DEMAND(phi.size() == 2);
    DRAKE_DEMAND(indices_A.size() == 2);
    DRAKE_DEMAND(indices_B.size() == 2);
    const double dist_link5 = std::min(phi[0], phi[1]);

    const double distance_tol = 1e-6;
    EXPECT_LT(std::min(dist_link5, std::min(dist_link6, dist_link7)),
              distance_tol);

    // Update t.
    t += dt;
  }
}

// Check that velocity at the contact point remains sufficiently close to zero.
TEST_F(ControllerTest, ZeroVelocityAtContact2) {
  // Get the plan.
  const ManipulationPlan& plan = controller_->plan();

  // Advance time, finding a point at which contact is desired.
  const double dt = 1e-3;
  double t = 0.0;
  const double t_final = plan.get_end_time();
  VectorXd q(nq_ball_ + nq_robot_), v(nv_ball_ + nqd_robot_);
  while (true) {
    if (plan.IsContactDesired(t)) {
      SetStates(t, &q, &v);
      break;
    }

    // No contact desired.
    t += dt;
    ASSERT_LT(t, t_final);
  }

  // Compute the kinematics cache.
  auto kinematics_cache =
      controller_->get_robot_and_ball_tree().doKinematics(q, v);

  // Get the necessary rigid bodies as non-constant.
  RigidBody<double>* link5_nc =
      const_cast<RigidBody<double>*>(&controller_->
          get_robot_link5_from_robot_and_ball_tree());
  RigidBody<double>* ball_nc =
      const_cast<RigidBody<double>*>(&controller_->
          get_ball_from_robot_and_ball_tree());

  // Construct the point of contact.
  drake::multibody::collision::PointPair<double> contact;
  contact.distance = 0;
  contact.normal = Vector3d(0, 0, 1);
  contact.elementA = *link5_nc->collision_elements_begin();
  contact.elementB = *ball_nc->collision_elements_begin();

  // Get the *planned* point of contact in the world.
  // Set the contact point to an arbitrary point in the world.
  const Vector3d p = plan.GetContactKinematics(t).head(3);

  // Set the contact point in the body frames.
  std::vector<drake::multibody::collision::PointPair<double>> contacts = {
      contact
  };
  const int body_a_index = contacts.front().
      elementA->get_body()->get_body_index();
  const int body_b_index = contacts.front().
      elementB->get_body()->get_body_index();
  contacts.front().ptA = kinematics_cache.get_element(body_a_index).
      transform_to_world.inverse(Eigen::Isometry) * p;
  contacts.front().ptB = kinematics_cache.get_element(body_b_index).
      transform_to_world.inverse(Eigen::Isometry) * p;

  // Construct the Jacobian matrices using the controller function.
  MatrixXd N, S, T, Ndot, Sdot, Tdot;
  controller_->ConstructJacobians(contacts, kinematics_cache,
                                  &N, &S, &T, &Ndot, &Sdot, &Tdot);
  EXPECT_EQ(Ndot.rows(), contacts.size());
  EXPECT_EQ(Ndot.cols(), 1);

  // Verify that the velocity at the contact point is approximately zero.
  const double zero_velocity_tol = 1e-12;
  const VectorXd Nv = N * v;
  const VectorXd Sv = S * v;
  const VectorXd Tv = T * v;
  ASSERT_EQ(contacts.size(), Nv.size());
  ASSERT_EQ(contacts.size(), Sv.size());
  ASSERT_EQ(contacts.size(), Tv.size());
  EXPECT_LT(Nv.norm(), zero_velocity_tol);
  EXPECT_LT(Sv.norm(), zero_velocity_tol);
  EXPECT_LT(Tv.norm(), zero_velocity_tol);
}

}  // namespace iiwa_soccer
}  // namespace examples
}  // namespace drake
