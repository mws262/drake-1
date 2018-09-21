#pragma once

#include "drake/examples/iiwa_soccer/manipulation_plan.h"
#include "drake/systems/framework/output_port.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/multibody/kinematics_cache.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/systems/framework/event.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/framework/vector_system.h"
#include "drake/systems/framework/continuous_state.h"

namespace drake {
namespace examples {
namespace iiwa_soccer {

class Controller : public systems::LeafSystem<double> {
  friend class ControllerTest_JacobianConstruction_Test;
  friend class ControllerTest_ZeroVelocityAtContact_Test;
  friend class ControllerTest_ZeroVelocityAtContact2_Test;

 public:

  // Constructor with LCM. Will draw target point if publish is on.
  Controller(const RigidBodyTree<double>& robot_and_ball_tree,
             const RigidBodyTree<double>& robot_tree,
             const Vector3<double>& k_p,
             const Vector3<double>& k_d,
             const VectorX<double>& joint_kp,
             const VectorX<double>& joint_ki,
             const VectorX<double>& joint_kd,
             drake::lcm::DrakeLcm& lcm) :
      Controller(robot_and_ball_tree, robot_tree, k_p, k_d,
                 joint_kp, joint_ki, joint_kd) {

    draw_status_ = true;
    lcm_ = &lcm;
    LoadPlans();
  }

  // Constructor without lcm. Draws nothing.
  Controller(const RigidBodyTree<double>& robot_and_ball_tree,
             const RigidBodyTree<double>& robot_tree,
             const Vector3<double>& k_p,
             const Vector3<double>& k_d,
             const VectorX<double>& joint_kp,
             const VectorX<double>& joint_ki,
             const VectorX<double>& joint_kd) :
      command_output_size_(robot_tree.get_num_actuators()),
      robot_and_ball_tree_(robot_and_ball_tree), robot_tree_(robot_tree),
      k_p_(k_p), k_d_(k_d),
      joint_kp_(joint_kp), joint_ki_(joint_ki), joint_kd_(joint_kd) {

    LoadPlans();
    this->DeclareContinuousState(nq_robot_); // For integral control state.
    input_port_index_estimated_robot_q_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nq_robot_)).get_index();
    input_port_index_estimated_robot_qd_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nqd_robot_)).get_index();
    input_port_index_estimated_ball_q_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nq_ball_)).get_index();
    input_port_index_estimated_ball_v_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nv_ball_)).get_index();
    this->DeclareVectorOutputPort(
        systems::BasicVector<double>(command_output_size_),
        &Controller::DoControlCalc); // Output 0.
  }

  const systems::OutputPort<double>& get_output_port_control() const {
    return this->get_output_port(0);
  }

  const systems::InputPort<double>& get_input_port_estimated_robot_q()
      const {
    return this->get_input_port(input_port_index_estimated_robot_q_);
  }

  const systems::InputPort<double>& get_input_port_estimated_robot_qd()
      const {
    return this->get_input_port(input_port_index_estimated_robot_qd_);
  }

  const systems::InputPort<double>& get_input_port_estimated_ball_q()
      const {
    return this->get_input_port(input_port_index_estimated_ball_q_);
  }

  const systems::InputPort<double>& get_input_port_estimated_ball_v()
      const {
    return this->get_input_port(input_port_index_estimated_ball_v_);
  }

  const ManipulationPlan& plan() const { return plan_; }

  void DoPublish(
      const systems::Context<double>& context,
      const std::vector<const systems::PublishEvent<double>*>& events) const;
  void set_integral_value(
      systems::Context<double>* context, const Eigen::VectorXd& qint) const;
  Eigen::VectorXd get_integral_value(
      const systems::Context<double>& context) const;
  const RigidBodyTree<double>& get_robot_tree() const {
    return robot_tree_; }
  const RigidBodyTree<double>& get_robot_and_ball_tree() const {
    return robot_and_ball_tree_;
  }
  int get_robot_velocity_start_index_in_v() const;
  int get_ball_velocity_start_index_in_v() const;
  int get_robot_position_start_index_in_x() const;
  int get_ball_position_start_index_in_q() const;
  int get_robot_position_start_index_in_q() const {
    return get_robot_position_start_index_in_x();
  }
  const RigidBody<double>& get_world_from_robot_and_ball_tree() const;
  const RigidBody<double>& get_ball_from_robot_and_ball_tree() const;
  const RigidBody<double>& get_robot_from_robot_and_ball_tree() const;
  const RigidBody<double>& get_robot_link5_from_robot_and_ball_tree() const;
  const RigidBody<double>& get_robot_link6_from_robot_and_ball_tree() const;
  const RigidBody<double>& get_robot_link7_from_robot_and_ball_tree() const;

 private:
  Eigen::VectorXd TransformVToQdot(
      const Eigen::VectorXd& q, const Eigen::VectorXd& v) const;
  void DoCalcTimeDerivatives(
      const systems::Context<double>& context,
      systems::ContinuousState<double>* derivatives) const;
  void DoControlCalc(const systems::Context<double>& context,
                     systems::BasicVector<double>* const output) const;
  void ConstructJacobians(
      const std::vector<drake::multibody::collision::PointPair<double>>&
          contacts,
      const KinematicsCache<double>& kinematics_cache,
      Eigen::MatrixXd* N, Eigen::MatrixXd* S, Eigen::MatrixXd* T,
      Eigen::MatrixXd* Ndot_v, Eigen::MatrixXd* Sdot_v,
      Eigen::MatrixXd* Tdot_v) const;
  void LoadPlans();
  Eigen::VectorXd get_all_q(const systems::Context<double>& context) const;
  Eigen::VectorXd get_all_v(const systems::Context<double>& context) const;
  Eigen::VectorXd ComputeTorquesForContactDesiredButNoContact(
      const systems::Context<double>& context) const;
  Eigen::VectorXd ComputeTorquesForContactDesiredAndContacting(
      const systems::Context<double>& context,
      const std::vector<drake::multibody::collision::PointPair<double>>&) const;
  Eigen::VectorXd ComputeTorquesForContactNotDesired(
      const systems::Context<double>& context) const;
  Eigen::VectorXd get_robot_q(const systems::Context<double>& context) const {
    return this->EvalVectorInput(
        context, input_port_index_estimated_robot_q_)->CopyToVector();
  }
  std::vector<drake::multibody::collision::PointPair<double>> FindContacts(
      const KinematicsCache<double>& kinematics_cache) const;

  Eigen::VectorXd get_robot_qd(const systems::Context<double>& context) const {
    return this->EvalVectorInput(
        context, input_port_index_estimated_robot_qd_)->CopyToVector();
  }

  Eigen::VectorXd get_ball_q(const systems::Context<double>& context) const {
    return this->EvalVectorInput(
        context, input_port_index_estimated_ball_q_)->CopyToVector();
  }

  Eigen::VectorXd get_ball_v(const systems::Context<double>& context) const {
    return this->EvalVectorInput(
        context, input_port_index_estimated_ball_v_)->CopyToVector();
  }


  const int nq_robot_ = 7;
  const int nqd_robot_ = 7;
  const int nq_ball_ = 7;
  const int nv_ball_ = 6;
  int input_port_index_estimated_robot_q_{-1};
  int input_port_index_estimated_robot_qd_{-1};
  int input_port_index_estimated_ball_q_{-1};
  int input_port_index_estimated_ball_v_{-1};
  const int command_output_size_; // number of robot motor torques.
  const RigidBodyTree<double>& robot_and_ball_tree_;
  const RigidBodyTree<double>& robot_tree_;
  const Vector3<double> k_p_;
  const Vector3<double> k_d_;
  lcm::DrakeLcm* lcm_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> joint_kp_,
      joint_ki_, joint_kd_;

  // If set to 'true', the controller will output visual debugging / logging
  // info through LCM messages.
  bool draw_status_ = false;

  // The plan for controlling the robot.
  ManipulationPlan plan_;
};


}  // namespace acrobot
}  // namespace examples
}  // namespace drake