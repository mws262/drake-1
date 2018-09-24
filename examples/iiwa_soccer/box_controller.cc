#include <Eigen/LU>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/common/sorted_pair.h"
#include "drake/examples/iiwa_soccer/controller.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/orthonormal_basis.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
//#include "drake/multibody/ik_options.h"
//#include "drake/multibody/rigid_body_constraint.h"
//#include "drake/multibody/rigid_body_ik.h"
#include "drake/multibody/rigid_body_plant/frame_visualizer.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

namespace drake {
namespace examples {
namespace iiwa_soccer {

using drake::geometry::SceneGraph;
using drake::math::RollPitchYaw;
using drake::math::RotationMatrix;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::InputPortDescriptor;
using drake::systems::OutputPort;
using drake::systems::PublishEvent;
using drake::systems::ContinuousState;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix3Xd;

// Loads all plans into the controller.
void BoxController::LoadPlans() {
  // Read in the plans for the robot.
  plan_.ReadRobotQQdotAndQddot(
      "examples/iiwa_soccer/plan/joint_timings_fit.mat",
      "examples/iiwa_soccer/plan/joint_angle_fit.mat",
      "examples/iiwa_soccer/plan/joint_vel_fit.mat",
      "examples/iiwa_soccer/plan/joint_accel_fit.mat");

  // Read in the plans for the point of contact.
  plan_.ReadContactPoint("examples/iiwa_soccer/plan/contact_pt_timings.mat",
      "examples/iiwa_soccer/plan/contact_pt_positions.mat",
      "examples/iiwa_soccer/plan/contact_pt_velocities.mat");

  // Read in the plans for the ball kinematics.
  plan_.ReadBallQVAndVdot(
      "examples/iiwa_soccer/plan/ball_timings.mat",
      "examples/iiwa_soccer/plan/ball_com_positions.mat",
      "examples/iiwa_soccer/plan/ball_quats.mat",
      "examples/iiwa_soccer/plan/ball_com_velocities.mat",
      "examples/iiwa_soccer/plan/ball_omegas.mat",
      "examples/iiwa_soccer/plan/ball_com_accelerations.mat",
      "examples/iiwa_soccer/plan/ball_alphas.mat",
      "examples/iiwa_soccer/plan/contact_status.mat");
}

// Constructs the Jacobian matrices.
void BoxController::ConstructJacobians(
    const Context<double>& context,
    const std::vector<geometry::PenetrationAsPointPair<double>>& contacts,
    MatrixXd* N, MatrixXd* S, MatrixXd* T,
    MatrixXd* Ndot_v, MatrixXd* Sdot_v, MatrixXd* Tdot_v) const {

  // Get the tree.
  const auto tree = robot_and_ball_plant_.model(); 

  // Get the numbers of contacts and generalized velocities.
  const int nc = static_cast<int>(contacts.size());

  // Set the number of generalized velocities.
  const int nv = tree.num_velocities(); 

  // Resize the matrices.
  N->resize(nc, nv);
  S->resize(nc, nv);
  T->resize(nc, nv);
  Ndot_v->resize(nc, 1);
  Sdot_v->resize(nc, 1);
  Tdot_v->resize(nc, 1);

  // Get the two body indices.
  for (int i = 0; i < nc; ++i) {
    const auto& point_pair = contacts[i];

    // Get the surface normal in the world frame.
    const Vector3d& n_BA_W = point_pair.nhat_BA_W:

    // TODO: get the two bodies. 
    const GeometryId geometryA_id = point_pair.id_A;
    const GeometryId geometryB_id = point_pair.id_B;
    const Body<double>& bodyA = tree.get_body(body_A_index);
    const BOdy<double>& bodyB = tree.get_body(body_B_index);

    // The reported point on A's surface (As) in the world frame (W).
    const Vector3d& p_WAs = point_pair.p_WCa;

    // The reported point on B's surface (Bs) in the world frame (W).
    const Vector3d& p_WBs = point_pair.p_WCb;

    // Get the point of contact in the world frame.
    const Vector3d p_W = (p_WAs + p_WBs) * 0.5;

    // Get the geometric Jacobian for the velocity of the contact point
    // as moving with Body A.
    MatrixXd J_WAc(3, nv);
    tree.CalcPointsGeometricJacobianExpressedInWorld(
        context, bodyA.body_frame(), p_W, &J_WAc); 

    // Get the geometric Jacobian for the velocity of the contact point
    // as moving with Body B.
    MatrixXd J_WBc(3, nv);
    tree.CalcPointsGeometricJacobianExpressedInWorld(
        context, bodyB.body_frame(), p_W, &J_WBc); 

    // Compute the linear components of the Jacobian.
    const MatrixXd J = J_WAc - J_WBc;

    // Compute an orthonormal basis using the contact normal.
    const int kXAxisIndex = 0, kYAxisIndex = 1, kZAxisIndex = 2;
    auto R_WC = math::ComputeBasisFromAxis(kXAxisIndex, n_BA_W);
    const Vector3d t1_BA_W = R_WC.col(kYAxisIndex);
    const Vector3d t2_BA_W = R_WC.col(kZAxisIndex);

    // Set N, S, and T.
    N->row(i) = n_BA_W.transpose() * J;
    S->row(i) = t1_BA_W.transpose() * J;
    T->row(i) = t2_BA_W.transpose() * J;

    // TODO: Set Ndot_v, Sdot_v, Tdot_v properly.
    Ndot_v->row(i).setZero(); // = n_BA_W.transpose() * Jdot_v;
    Sdot_v->row(i).setZero(); // = t1_BA_W.transpose() * Jdot_v;
    Tdot_v->row(i).setZero(); // = t2_BA_W.transpose() * Jdot_v;
  }
}

// TODO: What should the box be doing when it is not supposed to make contact?
// Computes the control torques when contact is not desired.
VectorXd BoxController::ComputeTorquesForContactNotDesired(
    const Context<double>& context) const {
  // Get the desired robot acceleration.
  const VectorXd q_robot_des = plan_.GetRobotQQdotAndQddot(
      context.get_time()).head(nq_robot_);
  const VectorXd qdot_robot_des = plan_.GetRobotQQdotAndQddot(
      context.get_time()).segment(nq_robot_, nv_robot_);
  const VectorXd qddot_robot_des = plan_.GetRobotQQdotAndQddot(
      context.get_time()).tail(nv_robot_);

  // Get the robot current generalized position and velocity.
  const VectorXd q_robot = get_robot_q(context);
  const VectorXd qd_robot = get_robot_qd(context);

  // Set qddot_robot_des using error feedback.
  const VectorXd qddot = qddot_robot_des +
      joint_kp_ * (q_robot_des - q_robot) +
      joint_ki_ * get_integral_value(context) +
      joint_kd_ * (qdot_robot_des - qd_robot);

  // Set the state in the robot context to q_robot and qd_robot. 
  VectorX<double> x = mbp_robot_.model().get_mutable_multibody_state_vector(
      robot_context_.get());
  DRAKE_DEMAND(x.size() == q_robot.size() + qd_robot.size()); 
  x.head(q_robot.size()) = q_robot;
  x.tail(qd_robot.size()) = qd_robot;

  // Get the generalized inertia matrix.
  MatrixXd M;
  mbp_robot_.model().CalcMassMatrixViaInverseDynamics(*robot_context_, &M);
  Eigen::LLT<MatrixXd> lltM(M);
  DRAKE_DEMAND(lltM.info() == Eigen::Success);

  // Compute the contribution from force elements.
  const auto& robot_tree = mbp_robot_.model();
  multibody::MultibodyForces<double> link_wrenches(robot_tree);
  PositionKinematicsCache<double> pcache(robot_tree.get_topology());
  VelocityKinematicsCache<double> vcache(robot_tree.get_topology());
  robot_tree.CalcPositionKinematicsCache(*robot_context_, &pcache);
  robot_tree.CalcVelocityKinematicsCache(*robot_context_, pcache, &vcache);

  // Compute the external forces.
  const VectorXd fext = -robot_tree.CalcInverseDynamics(
      *robot_context_, VectorXd::Zero(nv_robot_), link_wrenches);

  // Compute inverse dynamics.
  return M * qddot - fext;
}

// Computes the control torques when contact is desired and the robot and the
// ball are *not* in contact.
VectorXd BoxController::ComputeTorquesForContactDesiredButNoContact(
    const Context<double>& context) const {
  // Get the generalized positions for the robot and the ball.
  const VectorXd q0 = get_all_q(context);
  VectorXd v0 = get_all_v(context);

  // Get the robot tree.
  const auto& robot_tree = mbp_robot_.model();

  // TODO: Set the joint velocities for the robot to zero.
  v0.segment(get_robot_velocity_start_index_in_v(), nv_robot_).setZero();

  // Transform the velocities to time derivatives of generalized
  // coordinates.
  VectorXd qdot0;
  mbp_both_.MapVelocityToQDot(context, v0, &qdot0);

  // TODO(edrumwri): turn this system into an actual discrete system.
  const double control_freq = 100.0;  // 100 Hz.
  const double dt = 1.0/control_freq;

  // Get the estimated position of the ball and the robot at the next time
  // step using a first order approximation to position and the current
  // velocities.
  const VectorXd q1 = q0 + dt * qdot0;

  // TODO: Update the context to use configuration q1 in the query.

  // Evaluate scene graph's output port, getting a SceneGraph reference.
  const geometry::QueryObject<double>& query_object = this->EvalAbstractInput(context, geometry_query_input_port_)->GetValue<geometry::QueryObject<double>>(); 

  // Get the closest points on the robot foot and the ball corresponding to q1
  // and v0.
  std::vector<SignedDistancePair<double>> closest_points =
      query_object.ComputeSignedDistancePairwiseClosestPoints();
  DRAKE_DEMAND(!closest_points.empty());
  int found_index = -1;
  const int num_data = static_cast<int>(closest_points.size());
  for (int i = 0; i < num_data; ++i) {
    // TODO: Get the two bodies in contact.

    // If the two bodies correspond to the foot (box) and the ball, mark the
    // found index and stop looping.
  }
  DRAKE_DEMAND(found_index >= 0);

  // Get the signed distance data structure. 
  auto& closest = closest_points[found_index];

  // TODO: Call A the body belonging to the robot. 
  if (something) {
    std::swap(closest.id_A, closest.id_B);
    std::swap(closest.p_ACa, closest.p_BCb);
  }   

  // Get the closest points on the bodies. They'll be in their respective body
  // frames. 
  const Vector3d& closest_Aa = closest.p_ACa; 
  const Vector3d& closest_Bb = closest.p_BCb;

  // TODO: transform the points in the body frames corresponding to q1 to the
  // world frame.

  // Get the vector from the closest point on the foot to the closest point
  // on the ball in the body frames.
  Vector3d linear_v_des = (closest_Bw - closest_Aw) / dt;

  // Get the robot current generalized position and velocity.
  const VectorXd q_robot = get_robot_q(context);
  const VectorXd qd_robot = get_robot_qd(context);

  // Set the state in the robot context to q_robot and qd_robot. 
  VectorX<double> x = robot_tree.get_mutable_multibody_state_vector(
      robot_context_.get());
  DRAKE_DEMAND(x.size() == q_robot.size() + qd_robot.size()); 
  x.head(q_robot.size()) = q_robot;
  x.tail(qd_robot.size()) = qd_robot;

  // Get the geometric Jacobian for the velocity of the closest point on the
  // robot as moving with the robot Body A.
  MatrixXd J_WAc(3, nv_robot_);
  robot_tree.CalcPointsGeometricJacobianExpressedInWorld(
      *robot_context_, body_A.body_frame(), closest_Aw, &J_WAc); 

  // Set the rigid body constraints.
  /*
  const int num_constraints = 1;
  WorldPositionConstraint constraint();

  // Set the inverse kinematics options.
  IKoptions ik_options;

  // Compute the desired robot configuration.
  VectorXd q_robot_des = q_robot;
  RigidBodyTree<double>* robot_tree_nc =
      const_cast<RigidBodyTree<double>*>(&robot_tree_);
  int info;
  std::vector<std::string> infeasible_constraint;
  inverseKin(robot_tree_nc, q_robot_des, q_robot_des,
             num_constraints, constraint_array, ik_options,
             &q_robot_des, &info, &infeasible_constraint);

  // Just use the current configuration if not successful.
  if (info > 6)
    q_robot_des = q_robot;
  */
  VectorXd q_robot_des = q_robot;

  // Use resolved-motion rate control to determine the robot velocity that
  // would be necessary to realize the desired end-effector velocity.
  Eigen::JacobiSVD<MatrixXd> svd(
      J, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const VectorXd qdot_robot_des = svd.solve(linear_v_des);

  // Set qddot_robot_des using purely error feedback.
  const VectorXd qddot =
      joint_kp_ * (q_robot_des - q_robot) +
          joint_kd_ * (qdot_robot_des - qd_robot);

  // Get the generalized inertia matrix.
  MatrixXd M;
  robot_tree.CalcMassMatrixViaInverseDynamics(*robot_context_, &M);
  Eigen::LLT<MatrixXd> lltM(M);
  DRAKE_DEMAND(lltM.info() == Eigen::Success);

  // Compute the contribution from force elements.
  multibody::MultibodyForces<double> link_wrenches(robot_tree);
  PositionKinematicsCache<double> pcache(robot_tree.get_topology());
  VelocityKinematicsCache<double> vcache(robot_tree.get_topology());
  robot_tree.CalcPositionKinematicsCache(*robot_context_, &pcache);
  robot_tree.CalcVelocityKinematicsCache(*robot_context_, pcache, &vcache);

  // Compute the external forces.
  const VectorXd fext = -robot_tree.CalcInverseDynamics(
      *robot_context_, VectorXd::Zero(nv_robot_), link_wrenches);

  // Compute inverse dynamics.
  return M * qddot - fext;
}

// Computes the control torques when contact is desired and the robot and the
// ball are in contact.
VectorXd BoxController::ComputeTorquesForContactDesiredAndContacting(
    const Context<double>& context,
    const std::vector<geometry::PenetrationAsPointPair<double>>& contacts) const {
  // ***************************************************************
  // Note: This code is specific to this example/system.
  // ***************************************************************

  // Get the number of generalized positions, velocities, and actuators.
  const int nv = robot_and_ball_tree_.get_num_velocities();
  DRAKE_DEMAND(nv == nv_robot_ + nv_ball_);
  const int num_actuators = robot_and_ball_tree_.get_num_actuators();
  DRAKE_DEMAND(num_actuators == nv_robot_);

  // The starting index of the ball in the generalized velocities array. Note
  // TwanCode, which uses angular velocities first in the array and linear
  // velocities second.
  const int robot_velocity_start_index_in_v =
      get_robot_velocity_start_index_in_v();
  const int ball_linear_velocity_start_index_in_v =
      get_ball_velocity_start_index_in_v() + 3;

  // Get the generalized positions and velocities.
  const VectorXd q = get_all_q(context);
  const VectorXd v = get_all_v(context);

  // Construct the actuation matrix.
  MatrixXd B = MatrixXd::Zero(nv, num_actuators);
  for (int i = 0; i < num_actuators; ++i)
    B(robot_velocity_start_index_in_v + i, i) = 1.0;

  // Get the generalized inertia matrix.
  MatrixXd M;
  mbp_robot_.model().CalcMassMatrixViaInverseDynamics(*robot_context_, &M);
  Eigen::LLT<MatrixXd> lltM(M);
  DRAKE_DEMAND(lltM.info() == Eigen::Success);

  // Compute the contribution from force elements.
  const auto& robot_tree = mbp_robot_.model();
  multibody::MultibodyForces<double> link_wrenches(robot_tree);
  PositionKinematicsCache<double> pcache(robot_tree.get_topology());
  VelocityKinematicsCache<double> vcache(robot_tree.get_topology());
  robot_tree.CalcPositionKinematicsCache(*robot_context_, &pcache);
  robot_tree.CalcVelocityKinematicsCache(*robot_context_, pcache, &vcache);

  // Compute the external forces.
  const VectorXd fext = -robot_tree.CalcInverseDynamics(
      *robot_context_, VectorXd::Zero(nv_robot_), link_wrenches);

  // TODO(edrumwri): Check whether desired ball acceleration is in the right
  // format to match with layout of M.

  // Get the desired ball acceleration.
  const VectorXd vdot_ball_des = plan_.GetBallQVAndVdot(context.get_time()).
      tail(nv_ball_);

  // Construct the weighting matrix.
  MatrixXd P = MatrixXd::Zero(nv_ball_, nv);
  const int ball_linear_velocity_start_index_in_ball_v = 3;
  for (int i = 0; i < 3; ++i) {
    const int index = ball_linear_velocity_start_index_in_v + i;
    P(ball_linear_velocity_start_index_in_ball_v + i, index) = 1.0;
  }

  // Construct the Jacobians and the Jacobians times the velocity.
  MatrixXd N, S, T;
  MatrixXd Ndot_v, Sdot_v, Tdot_v;
  ConstructJacobians(contacts, &N, &S, &T, &Ndot_v, &Sdot_v, &Tdot_v);

  // Get the Jacobians at the point of contact: N, S, T, and construct Z and
  // Zdot_v.
  MatrixXd Z(N.rows() * 3, N.cols());
  Z.topRows(N.rows()) = N;
  Z.middleRows(N.rows(), N.rows()) = S;
  Z.bottomRows(N.rows()) = T;

  // Set the time-derivatives of the Jacobians times the velocity.
  MatrixXd Zdot_v(Ndot_v.rows() * 3, Ndot_v.cols());
  Zdot_v.topRows(Ndot_v.rows()) = Ndot_v;
  Zdot_v.middleRows(Ndot_v.rows(), Ndot_v.rows()) = Sdot_v;
  Zdot_v.bottomRows(Ndot_v.rows()) = Tdot_v;
  DRAKE_DEMAND(Zdot_v.cols() == 1);

  // Primal variables are motor torques and contact force magnitudes.
  const int nc = static_cast<int>(contacts.size());
//  const int nprimal = num_actuators + nc * 3;
const int nprimal = num_actuators + nc;

  // Dual variables (Lagrange multipliers) correspond to number of linear
  // constraint equations.
//  const int ndual = nc * 3;
  const int ndual = nc;

  // Construct the matrices necessary to construct the Hessian.
//  MatrixXd D(nv, num_actuators + nc * 3);
  MatrixXd D(nv, num_actuators + nc);
  D.setZero();
  D.topLeftCorner(B.rows(), B.cols()) = B;
//  D.bottomRightCorner(Z.cols(), Z.rows()) = Z.transpose();
  D.bottomRightCorner(N.cols(), N.rows()) = N.transpose();

  // Set the Hessian matrix for the QP.
  MatrixXd H = D.transpose() * lltM.solve(P.transpose()) * P *
      lltM.solve(D);
  Eigen::LDLT<MatrixXd> lltH;

  // Verify that the Hessian is positive semi-definite.
  H += Eigen::MatrixXd::Identity(H.rows(), H.cols()) * 1e-8;
  DRAKE_ASSERT(lltH.compute(H).info() == Eigen::Success);

  // Compute the linear terms.
  const VectorXd c = D.transpose() * lltM.solve(P.transpose()) * (
      -vdot_ball_des + P * lltM.solve(fext));

  // Set the affine constraint matrix.
//  const MatrixXd A = Z * lltM.solve(D);
//  const VectorXd b = -Z * lltM.solve(fext) - Zdot_v;
  const MatrixXd A = N * lltM.solve(D);
  const VectorXd b = -N * lltM.solve(fext) - Ndot_v;
  DRAKE_DEMAND(b.rows() == ndual);

  // Prepare to solve the QP using the direct solution to the KKT system.
  MatrixXd K(nprimal + ndual, nprimal + ndual);
  K.block(0, 0, nprimal, nprimal) = H;
  K.block(0, nprimal, nprimal, ndual) = -A.transpose();
  K.block(nprimal, ndual, ndual, nprimal).setZero();
  K.block(nprimal, 0, ndual, nprimal) = A;

  // Set the right hand side for the KKT solutoin.
  VectorXd rhs(nprimal + ndual);
  rhs.segment(0, nprimal) = -c;
  rhs.segment(nprimal, ndual) = b;

  // Solve the KKT system.
  const Eigen::PartialPivLU<MatrixXd> lu(K);
  const VectorXd z = lu.solve(rhs);

  // Verify that the solution is reasonably accurate.
  const double tol = 1e-8;
  const double soln_err = (K * z - rhs).norm();
  DRAKE_DEMAND(soln_err < tol);

/*  // Check whether the computed normal contact forces are tensile.
  const VectorXd cf = z.tail(nc * 3);
  bool tensile = false;
  for (int i = 0; i < cf.size(); i += 3) {
    if (cf[i] < 0.0) {
      tensile = true;
      break;
    }
  }
*/
const VectorXd cf = z.tail(nc);
  // Output some logging information.
  const VectorXd vdot = lltM.solve(D*z.head(nprimal) + fext);
  const VectorXd P_vdot = P * vdot;
  std::cout << "N * v: " << (N * v).transpose() << std::endl;
  std::cout << "S * v: " << (S * v).transpose() << std::endl;
  std::cout << "T * v: " << (T * v).transpose() << std::endl;
  std::cout << "Ndot * v: " << Ndot_v << std::endl;
  std::cout << "Zdot * v: " << Zdot_v << std::endl;
  std::cout << "fext: " << fext.transpose() << std::endl;
  std::cout << "M: " << std::endl << M << std::endl;
  std::cout << "P: " << std::endl << P << std::endl;
  std::cout << "D: " << std::endl << D << std::endl;
  std::cout << "B: " << std::endl << B << std::endl;
  std::cout << "N: " << std::endl << N << std::endl;
  std::cout << "Z: " << std::endl << Z << std::endl;
  std::cout << "contact forces: " << cf.transpose() << std::endl;
  std::cout << "vdot: " << vdot.transpose() << std::endl;
  std::cout << "vdot (desired): " << vdot_ball_des.transpose() << std::endl;
  std::cout << "P * vdot: " << P_vdot.transpose() << std::endl;
  std::cout << "torque: " << z.head(nv_robot_).transpose() << std::endl;

  // First nv_robot_ primal variables are the torques.
  return z.head(nv_robot_);
}

// Gets the vector of contacts.
std::vector<geometry::PenetrationAsPointPair<double>> BoxController::FindContacts() const {
  // Evaluate scene graph's output port, getting a SceneGraph reference.
  const geometry::QueryObject<double>& query_object = this->EvalAbstractInput(context, geometry_query_input_port_)->GetValue<geometry::QueryObject<double>>(); 

  // Determine the set of contacts.
  std::vector<geometry::PenetrationAsPointPair<double>> contacts = query_object.ComputePointPairPenetration();

  // Get the ball body and foot bodies.
  const auto ball_body = &get_ball_from_robot_and_ball_tree();
  const auto box_body = &get_box_from_robot_and_ball_tree();
  const auto world_body = &get_world_from_robot_and_ball_tree();

  // Make sorted pairs to check.
  const auto ball_box_pair = make_sorted_pair(ball_body, box_body);

  // Remove contacts between all but the robot foot and the ball and the
  // ball and the ground.
  for (int i = 0; i < static_cast<int>(contacts.size()); ++i) {
    // TODO: Get the two bodies involved in the contact.
    const auto body_a = contacts[i].elementA->get_body();
    const auto body_b = contacts[i].elementB->get_body();
    const auto body_a_b_pair = make_sorted_pair(body_a, body_b);
    if (body_a_b_pair != ball_box_pair) {
      contacts[i] = contacts.back();
      contacts.pop_back();
      --i;
    } else {
//      std::cout << "Contact detected between " << body_a->get_name() <<
//                " and " << body_b->get_name() << std::endl;
    }
  }

  return contacts;
}

/**
 *  Calculate what torques to apply to the joints.
 * @param context
 * @param output
 */
void BoxController::DoControlCalc(
    const Context<double>& context,
    BasicVector<double>* const output) const {
  // Determine whether we're in a contacting or not-contacting phase.
  const bool contact_desired = plan_.IsContactDesired(context.get_time());

  // Get the number of generalized positions, velocities, and actuators.
  const int nv = robot_and_ball_tree_.get_num_velocities();
  DRAKE_DEMAND(nv == nv_robot_ + nv_ball_);
  const int num_actuators = robot_and_ball_tree_.get_num_actuators();
  DRAKE_DEMAND(num_actuators == nv_robot_);

  // Get the generalized positions and velocities.
  VectorXd q = get_all_q(context);
  VectorXd v = get_all_v(context);

  // Compute tau.
  VectorXd tau;
  if (contact_desired) {
    // Find contacts.
    auto contacts = FindContacts();

    // Get the number of points of contact.
    const int nc = static_cast<int>(contacts.size());

    // Two cases: in the first, the robot and the ball are already in contact,
    // as desired. In the second, the robot desires to be in contact, but the
    // ball and the robot are not contacting: the robot must intercept the ball.
    if (nc >= 2) {
      tau = ComputeTorquesForContactDesiredAndContacting(context, contacts);
    } else {
      tau = ComputeTorquesForContactDesiredButNoContact(context);
    }
  } else {
    // No contact desired.
    tau = ComputeTorquesForContactNotDesired(context);
  }

  // Set the torque output.
  BasicVector<double> torque_out(tau);
  output->SetFrom(torque_out);
}

/// Gets the value of the integral term in the state.
VectorXd BoxController::get_integral_value(const Context<double>& context) const {
  return context.get_continuous_state_vector().CopyToVector();
}

/// Sets the value of the integral term in the state.
void BoxController::set_integral_value(
    Context<double>* context, const VectorXd& qint) const {
  DRAKE_DEMAND(qint.size() == nv_robot_);
  context->get_mutable_continuous_state_vector().SetFromVector(qint);
}

void BoxController::DoCalcTimeDerivatives(
    const Context<double>& context,
    ContinuousState<double>* derivatives) const {
  // Determine whether we're in a contacting or not-contacting phase.
  const bool contact_intended = plan_.IsContactDesired(context.get_time());

  if (contact_intended) {
    derivatives->get_mutable_vector().SetFromVector(VectorXd::Zero(nq_robot_));
  } else {
    // Get the desired robot configuration.
    const VectorXd q_robot_des = plan_.GetRobotQQdotAndQddot(
        context.get_time()).head(nq_robot_);

    // Get the current robot configuration.
    const auto x = dynamic_cast<const BasicVector<double>&>(
        context.get_continuous_state_vector()).get_value();
    const VectorXd q_robot = robot_and_ball_plant_.model().get_positions_from_array(robot_model_instance_, x);
    derivatives->get_mutable_vector().SetFromVector(q_robot_des - q_robot);
  }
}

/// Gets the ball body from the robot and ball tree.
const Body<double>& BoxController::get_ball_from_robot_and_ball_tree() const {
  return robot_and_ball_plant_.model().GetBodyByName("ball"); 
}

/// Gets the box link from the robot and ball tree.
const RigidBody<double>& BoxController::get_box_from_robot_and_ball_tree()
    const {
  return robot_and_ball_plant_.model().GetBodyByName("box"); 
}

/// Gets the world body from the robot and ball tree.
const RigidBody<double>& BoxController::get_world_from_robot_and_ball_tree()
    const {
  return robot_and_ball_plant_.model().world_body();
}

VectorXd BoxController::get_all_q(const Context<double>& context) const {
  const VectorXd robot_q = get_robot_q(context);
  const VectorXd ball_q = get_ball_q(context);
  VectorXd q(nq_ball_ + nq_robot_);

  // Sanity check.
  for (int i = 0; i < q.size(); ++i)
    q[i] = std::numeric_limits<double>::quiet_NaN();

// TODO: update this.
  q.segment(get_robot_position_start_index_in_q(), nq_robot_) = robot_q;
  q.segment(get_ball_position_start_index_in_q(), nq_ball_) = ball_q;

  // Sanity check.
  for (int i = 0; i < q.size(); ++i)
    DRAKE_DEMAND(!std::isnan(q[i]));

  return q;
}

Eigen::VectorXd BoxController::get_all_v(const Context<double>& context) const {
  const VectorXd robot_qd = get_robot_qd(context);
  const VectorXd ball_v = get_ball_v(context);
  VectorXd v(nv_ball_ + nv_robot_);

  // Sanity check.
  for (int i = 0; i < v.size(); ++i)
    v[i] = std::numeric_limits<double>::quiet_NaN();

// TODO: update this.
  v.segment(get_robot_velocity_start_index_in_v(), nv_robot_) = robot_qd;
  v.segment(get_ball_velocity_start_index_in_v(), nv_ball_) = ball_v;
  return v;
}


void BoxController::DoPublish(
    const Context<double>&,
    const std::vector<const PublishEvent<double>*>&) const {
  /*
  if (draw_status_ && (0.05 - std::fmod(context.get_time(), 0.05)) < 0.001) {
    // Draws the location of the controller's target.
    drake::lcmt_viewer_draw frame_msg{};
    frame_msg.timestamp = 0;
    frame_msg.num_links = 1;
    frame_msg.link_name.resize(1);
    frame_msg.robot_num.resize(1, 0);

    Eigen::Isometry3f pose;
    pose.setIdentity();

    Eigen::Quaternion<float> goal_quat =
        Eigen::Quaternion<float>(pose.linear());
    frame_msg.link_name[0] = "imped_controller_target";
    frame_msg.position.push_back({static_cast<float>(x_target(0)), static_cast<float>(x_target(1)),
                                  static_cast<float>(x_target(2))});
    frame_msg.quaternion.push_back(
        {goal_quat.w(), goal_quat.x(), goal_quat.y(), goal_quat.z()});

    const int num_bytes = frame_msg.getEncodedSize();
    const size_t size_bytes = static_cast<size_t>(num_bytes);
    std::vector<uint8_t> bytes(size_bytes);
    frame_msg.encode(bytes.data(), 0, num_bytes);
    lcm_->Publish("DRAKE_DRAW_FRAMES_T", bytes.data(), num_bytes, {});
  }
*/
}

}  // namespace acrobot
}  // namespace examples
}  // namespace drake
