#pragma once

#include "drake/examples/iiwa_soccer/manipulation_plan.h"
#include "drake/systems/framework/output_port.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/multibody/multibody_tree/multibody_tree.h"
#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"
#include "drake/systems/framework/event.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/systems/framework/vector_system.h"
#include "drake/systems/framework/continuous_state.h"

namespace drake {
namespace examples {
namespace iiwa_soccer {

class BoxController : public systems::LeafSystem<double> {
 public:

  // Constructor with LCM. Will draw target point if publish is on.
  BoxController(
      const multibody::multibody_plant::MultibodyPlant<double>&
          robot_and_ball_plant,
      const multibody::multibody_plant::MultibodyPlant<double>& robot_mbp,
      const Vector3<double>& k_p,
      const Vector3<double>& k_d,
      const VectorX<double>& joint_kp,
      const VectorX<double>& joint_ki,
      const VectorX<double>& joint_kd,
      drake::lcm::DrakeLcm& lcm) :
      BoxController(robot_and_ball_plant, robot_mbp, k_p, k_d,
                 joint_kp, joint_ki, joint_kd) {

    draw_status_ = true;
    lcm_ = &lcm;
    LoadPlans();
  }

  // Constructor without lcm. Draws nothing.
  BoxController(
      const multibody::multibody_plant::MultibodyPlant<double>&
          robot_and_ball_plant,
      const multibody::multibody_plant::MultibodyPlant<double>& robot_mbp,
      const Vector3<double>& k_p,
      const Vector3<double>& k_d,
      const VectorX<double>& joint_kp,
      const VectorX<double>& joint_ki,
      const VectorX<double>& joint_kd) :
      command_output_size_(robot_mbp.num_actuators()),
      robot_and_ball_plant_(robot_and_ball_plant),
      robot_mbp_(robot_mbp),
      k_p_(k_p),
      k_d_(k_d),
      joint_kp_(joint_kp),
      joint_ki_(joint_ki),
      joint_kd_(joint_kd) {
    LoadPlans();

    // Declare states and ports.
    this->DeclareContinuousState(nq_robot()); // For integral control state.
    input_port_index_estimated_robot_q_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nq_robot())).get_index();
    input_port_index_estimated_robot_qd_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nv_robot())).get_index();
    input_port_index_estimated_ball_q_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nq_ball())).get_index();
    input_port_index_estimated_ball_v_ = this->DeclareVectorInputPort(
        systems::BasicVector<double>(nv_ball())).get_index();
    this->DeclareVectorOutputPort(
        systems::BasicVector<double>(command_output_size_),
        &BoxController::DoControlCalc); // Output 0.

    // Create the necessary contexts.
    robot_context_ = robot_mbp_.CreateDefaultContext();
//    scenegraph_and_mbp_query_context_ = robot_and_ball_plant_.
//        CreateDefaultContext();

    // Get model instance indices from the composite plant.
    robot_instance_ = robot_and_ball_plant_.GetModelInstanceByName("robot");
    ball_instance_ = robot_and_ball_plant_.GetModelInstanceByName("ball");
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
  const multibody::Body<double>& get_world_from_robot_and_ball_tree() const;
  const multibody::Body<double>& get_ball_from_robot_and_ball_tree() const;
  const multibody::Body<double>& get_box_from_robot_and_ball_tree() const;

 private:
  Eigen::MatrixXd ConstructActuationMatrix() const;
  Eigen::MatrixXd ConstructWeightingMatrix() const;
  void UpdateRobotAndBallConfigurationForGeometricQueries(
      const VectorX<double>& q) const;
  Eigen::VectorXd TransformVToQdot(
      const Eigen::VectorXd& q, const Eigen::VectorXd& v) const;
  void DoCalcTimeDerivatives(
      const systems::Context<double>& context,
      systems::ContinuousState<double>* derivatives) const;
  void DoControlCalc(const systems::Context<double>& context,
                     systems::BasicVector<double>* const output) const;
  void ConstructJacobians(
      const systems::Context<double>& context,
      const std::vector<geometry::PenetrationAsPointPair<double>>& contacts,
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
      const std::vector<geometry::PenetrationAsPointPair<double>>&) const;
  Eigen::VectorXd ComputeTorquesForContactNotDesired(
      const systems::Context<double>& context) const;
  Eigen::VectorXd get_robot_q(const systems::Context<double>& context) const {
    return this->EvalVectorInput(
        context, input_port_index_estimated_robot_q_)->CopyToVector();
  }
  std::vector<geometry::PenetrationAsPointPair<double>> FindContacts() const;

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

  int nq_robot() const { return robot_mbp_.tree().num_positions(); }
  int nv_robot() const { return robot_mbp_.tree().num_velocities(); }
  int nq_ball() const {
      return robot_and_ball_plant_.tree().num_positions() - nq_robot();
  }
  int nv_ball() const {
      return robot_and_ball_plant_.tree().num_velocities() - nv_robot();
  }

  int input_port_index_estimated_robot_q_{-1};
  int input_port_index_estimated_robot_qd_{-1};
  int input_port_index_estimated_ball_q_{-1};
  int input_port_index_estimated_ball_v_{-1};
  const int command_output_size_; // number of robot motor torques.
  lcm::DrakeLcm* lcm_;

  // The plant holding everything: robot and ball.
  const multibody::multibody_plant::MultibodyPlant<double>&
      robot_and_ball_plant_;

  // A plant for the robot only.
  const multibody::multibody_plant::MultibodyPlant<double>& robot_mbp_;

  // A context for making queries with robot_mbp_.
  std::unique_ptr<systems::Context<double>> robot_context_;

  // A context for making contact queries with SceneGraph.
  std::unique_ptr<systems::Context<double>> scenegraph_and_mbp_query_context_;

  // A mapping from registered geometry indices to body indices.
  std::unordered_map<geometry::GeometryId, multibody::BodyIndex>
      geometry_id_to_body_index_;

  // PD gains.
  const Vector3<double> k_p_;
  const Vector3<double> k_d_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> joint_kp_,
      joint_ki_, joint_kd_;

  // Instance index for the robot in robot_and_ball_plant_.
  multibody::ModelInstanceIndex robot_instance_;

  // Instance index for the ball in robot_and_ball_plant_.
  multibody::ModelInstanceIndex ball_instance_;

  // Port for the SceneGraph query object.
  int geometry_query_input_port_{-1};

  // If set to 'true', the controller will output visual debugging / logging
  // info through LCM messages.
  bool draw_status_ = false;

  // The plan for controlling the robot.
  ManipulationPlan plan_;
};


}  // namespace acrobot
}  // namespace examples
}  // namespace drake