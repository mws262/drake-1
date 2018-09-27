/// @file
///
/// Kuka iiwa plays soccer. Wow!
///

#include <memory>
#include <vector>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/examples/iiwa_soccer/controller.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/sim_diagram_builder.h"
#include "drake/manipulation/util/world_sim_tree_builder.h"
#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"
#include "drake/multibody/multibody_tree/uniform_gravity_field_element.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/saturation.h"

DEFINE_double(simulation_sec, std::numeric_limits<double>::infinity(), "Number of seconds to simulate.");
DEFINE_double(target_realtime_rate, 1.0, "Playback speed x realtime");

DEFINE_double(kp, 60, "Cartesian kp for impedance control. Gets used for all xyz directions.");
DEFINE_double(kd, 30, "Cartesian kp for impedance control. Gets used for all xyz directions.");

namespace drake {
namespace examples {
namespace iiwa_soccer {
namespace {

using drake::geometry::SceneGraph;
using drake::lcm::DrakeLcm;
using drake::manipulation::util::ModelInstanceInfo;
using drake::manipulation::util::SimDiagramBuilder;
using drake::manipulation::util::WorldSimTreeBuilder;
using drake::multibody::Body;
using drake::multibody::multibody_plant::MultibodyPlant;
using drake::multibody::MultibodyTree;
using drake::multibody::parsing::AddModelFromSdfFile;
using drake::parsers::urdf::AddModelInstanceFromUrdfFileToWorld;
using drake::systems::ConstantVectorSource;
using drake::systems::Context;
using drake::systems::Demultiplexer;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::Multiplexer;
using drake::systems::Simulator;
using drake::systems::controllers::InverseDynamics;
using drake::trajectories::PiecewisePolynomial;
using Eigen::VectorXd;
using kuka_iiwa_arm::kIiwaArmNumJoints;
using kuka_iiwa_arm::kIiwaLcmStatusPeriod;
using kuka_iiwa_arm::IiwaStatusSender;
using kuka_iiwa_arm::IiwaCommandReceiver;

const char* armModelPath = //"drake/examples/iiwa_soccer/models/iiwa14_spheres_collision.urdf";
    "drake/manipulation/models/iiwa_description/sdf/"
        "iiwa14_no_collision.sdf";
const char* ballModelPath = "drake/examples/iiwa_soccer/models/soccer_ball.urdf";

const Eigen::Vector3d robot_base_location(0, 0, 0);

/**
 * Send a frame to be drawn over LCM.
 *
 * @param poses
 * @param name
 * @param dlcm
static void PublishFrames(std::vector<Eigen::Isometry3d> poses, std::vector<std::string> name, drake::lcm::DrakeLcm& dlcm) {
  drake::lcmt_viewer_draw frame_msg{};
  frame_msg.timestamp = 0;
  int32_t vsize = poses.size();
  frame_msg.num_links = vsize;
  frame_msg.link_name.resize(vsize);
  frame_msg.robot_num.resize(vsize, 0);

  for (size_t i = 0; i < poses.size(); i++) {
    Eigen::Isometry3f pose = poses[i].cast<float>();
    // Create a frame publisher
    Eigen::Vector3f goal_pos = pose.translation();
    Eigen::Quaternion<float> goal_quat =
        Eigen::Quaternion<float>(pose.linear());
    frame_msg.link_name[i] = name[i];
    frame_msg.position.push_back({goal_pos(0), goal_pos(1), goal_pos(2)});
    frame_msg.quaternion.push_back({goal_quat.w(), goal_quat.x(), goal_quat.y(), goal_quat.z()});
  }

  const int num_bytes = frame_msg.getEncodedSize();
  const size_t size_bytes = static_cast<size_t>(num_bytes);
  std::vector<uint8_t> bytes(size_bytes);
  frame_msg.encode(bytes.data(), 0, num_bytes);
  dlcm.Publish("DRAKE_DRAW_FRAMES", bytes.data(), num_bytes, {});

  drake::log()->info("Completed Frame Publishing");
}
 */


int DoMainControlled() {
  /******** Setup the world. ********/
  drake::lcm::DrakeLcm lcm; // Communications to robot or visualizer.

  systems::DiagramBuilder<double> builder;

  auto tree = std::make_unique<RigidBodyTree<double>>();

  // Construct the tree for simulation.
  AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow(armModelPath),
      multibody::joints::kFixed,
      tree.get());
  AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow(ballModelPath),
      multibody::joints::kQuaternion,
      tree.get());

  // Make ground part of tree.
  multibody::AddFlatTerrainToWorld(tree.get(), 100., 10.);

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Construct the multibody plant using both the Kuka and ball models.
  MultibodyPlant<double>& mb_plant = *builder.AddSystem<MultibodyPlant>();
  auto kuka_id_table = AddModelFromSdfFile(
      FindResourceOrThrow(armModelPath), &mb_plant, &scene_graph);
  auto ball_id_table = AddModelFromSdfFile(
      FindResourceOrThrow(ballModelPath), &mb_plant, &scene_graph);

  // Add gravity to the model.
  mb_plant.AddForceElement<multibody::UniformGravityFieldElement>(
      -9.81 * Vector3<double>::UnitZ());
  mb_plant.Finalize(&scene_graph);

/*
  RigidBodyPlant<double>* plant_ptr = builder.AddPlant(std::move(tree));
  systems::CompliantMaterial default_material;
  default_material.set_youngs_modulus(1e8)
      .set_dissipation(1.0)
      .set_friction(.02, .02);
  plant_ptr->set_default_compliant_material(default_material);
*/
  // Construct the tree for control.
  auto control_tree = std::make_unique<RigidBodyTree<double>>();
  AddModelInstanceFromUrdfFileToWorld(FindResourceOrThrow(armModelPath),
                                      multibody::joints::kFixed,
                                      control_tree.get());

  /******** Add systems. ********/

  // Gains in cartesian-land.
  Vector3<double> k_p;
  k_p << FLAGS_kp, FLAGS_kp, FLAGS_kp;
  Vector3<double> k_d;
  k_d << FLAGS_kd, FLAGS_kd, FLAGS_kd;

  // Joint gains for the robot.
  VectorXd joint_kp(7), joint_ki(7), joint_kd(7);
  joint_kp.setOnes() *= 10.0;
  joint_ki.setOnes() *= 0.1;
  joint_kd.setOnes() *= 1.0;

  auto controller = builder.AddSystem<Controller>(
      *tree, *control_tree, k_p, k_d,
      joint_kp, joint_ki, joint_kd, lcm);

  // Actuator saturation: no limits.
  double lim = 1e10;
  VectorX<double> min_lim(7);
  min_lim << -lim, -lim, -lim, -lim, -lim, -lim, -lim;
  VectorX<double> max_lim = -min_lim;
  auto saturation = builder.AddSystem<systems::Saturation>(min_lim, max_lim);

  // Construct the necessary demultiplexers.
  const int nq_ball = 7, nq_robot = 7, nv_ball = 6, nqd_robot = 7;
  auto robot_state_demuxer = builder.AddSystem<Demultiplexer<double>>(
      nq_robot + nqd_robot, nq_robot);
  auto ball_state_demuxer = builder.AddSystem<Demultiplexer<double>>(
      nq_ball + nv_ball, 1);

  // Construct the necessary mutliplexers.
  std::vector<int> ball_config_port_input_sizes = { 1, 1, 1, 1,  1, 1, 1 };
  std::vector<int> ball_vel_port_input_sizes = { 1, 1, 1, 1, 1, 1 };
  auto ball_q_muxer = builder.AddSystem<Multiplexer<double>>(
      ball_config_port_input_sizes);
  auto ball_v_muxer = builder.AddSystem<Multiplexer<double>>(
      ball_vel_port_input_sizes);

  // Constant source 2.
  /******** FULL IIWA INFO TO LCM **********/

  auto iiwa_status_pub = builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<lcmt_iiwa_status>("IIWA_STATUS", &lcm));
  iiwa_status_pub->set_name("iiwa_status_publisher");
  iiwa_status_pub->set_publish_period(kIiwaLcmStatusPeriod);
  auto iiwa_status_sender = builder.AddSystem<IiwaStatusSender>(7);
  iiwa_status_sender->set_name("iiwa_status_sender");
  auto iiwa_command_receiver = builder.AddSystem<IiwaCommandReceiver>(7);
  iiwa_command_receiver->set_name("iwwa_command_receiver");
  auto iiwa_command_sub = builder.AddSystem(systems::lcm::LcmSubscriberSystem::Make<lcmt_iiwa_command>("IIWA_COMMAND", &lcm));
  iiwa_command_sub->set_name("iiwa_command_subscriber");

  builder.Connect(iiwa_command_sub->get_output_port(), iiwa_command_receiver->get_input_port(0));
//  builder.Connect(plant_ptr->model_instance_state_output_port(kuka_id_table.at("iiwa14")), iiwa_status_sender->get_state_input_port());
  builder.Connect(mb_plant.get_continuous_state_output_port(kuka_id_table), iiwa_status_sender->get_state_input_port());
  builder.Connect(iiwa_command_receiver->get_output_port(0), iiwa_status_sender->get_command_input_port());
//  builder.Connect(plant_ptr->model_instance_state_output_port(kuka_id_table.at("iiwa14")), robot_state_demuxer->get_input_port(0));
//  builder.Connect(plant_ptr->model_instance_state_output_port(ball_id_table.at("soccer_ball")), ball_state_demuxer->get_input_port(0));
  builder.Connect(mb_plant.get_continuous_state_output_port(kuka_id_table), robot_state_demuxer->get_input_port(0));
  builder.Connect(mb_plant.get_continuous_state_output_port(ball_id_table), ball_state_demuxer->get_input_port(0));
  for (int i = 0; i < nq_ball; ++i)
    builder.Connect(ball_state_demuxer->get_output_port(i), ball_q_muxer->get_input_port(i));
  for (int i = 0; i < nv_ball; ++i)
    builder.Connect(ball_state_demuxer->get_output_port(nq_ball + i), ball_v_muxer->get_input_port(i));
  builder.Connect(ball_q_muxer->get_output_port(0), controller->get_input_port_estimated_ball_q());
  builder.Connect(ball_v_muxer->get_output_port(0), controller->get_input_port_estimated_ball_v());
  builder.Connect(robot_state_demuxer->get_output_port(0), controller->get_input_port_estimated_robot_q());
  builder.Connect(robot_state_demuxer->get_output_port(1), controller->get_input_port_estimated_robot_qd());
  builder.Connect(controller->get_output_port_control(), saturation->get_input_port());
  builder.Connect(saturation->get_output_port(), iiwa_status_sender->get_commanded_torque_input_port());
//  builder.Connect(saturation->get_output_port(), plant_ptr->model_instance_actuator_command_input_port(kuka_id_table.at("iiwa14")));
  builder.Connect(saturation->get_output_port(), mb_plant.get_actuation_input_port());
  builder.Connect(iiwa_status_sender->get_output_port(0), iiwa_status_pub->get_input_port());


  // Last thing before building the diagram; configure the system for
  // visualization.
  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);
  auto diagram = builder.Build();

  ///////////////////////////
  std::vector<Eigen::Isometry3d> poses_to_draw;
  Eigen::Isometry3d target_pose;
  target_pose.setIdentity();
  Eigen::Vector3d tarPos;

  // TODO(edrumwri): Add poses to draw.
//  PublishFrames(poses_to_draw, pose_names, lcm);

  ////////////////////////////////

  Simulator<double> simulator(*diagram);
  simulator.set_publish_at_initialization(false);
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);

  // Set the initial conditions, according to the plan.
  Context<double>& context = simulator.get_mutable_context();
  const ManipulationPlan& plan = controller->plan();
  const double t0 = 0;
  VectorXd q(nq_ball + nq_robot);
  VectorXd v(nv_ball + nqd_robot);
  const VectorXd q_robot = plan.GetRobotQQdotAndQddot(t0).head(nq_robot);
  const VectorXd qd_robot = plan.GetRobotQQdotAndQddot(t0).segment(
      nq_robot, nqd_robot);
  const VectorXd q_ball = plan.GetBallQVAndVdot(t0).head(nq_ball);
  const VectorXd v_ball = plan.GetBallQVAndVdot(t0).segment(nq_ball, nv_ball);
  q.segment(controller->get_robot_position_start_index_in_q(), nq_robot) =
      q_robot;
  q.segment(controller->get_ball_position_start_index_in_q(), nq_ball) = q_ball;
  v.segment(controller->get_robot_velocity_start_index_in_v(),
      nqd_robot) = qd_robot;
  v.segment(controller->get_ball_velocity_start_index_in_v(), nv_ball) = v_ball;
  context.get_mutable_continuous_state().get_mutable_generalized_position().SetFromVector(q);
  context.get_mutable_continuous_state().get_mutable_generalized_velocity().SetFromVector(v);

  simulator.reset_integrator<systems::RungeKutta2Integrator<double>>(*diagram, 1e-3, &context);
  simulator.Initialize();
  simulator.StepTo(FLAGS_simulation_sec);

  return 0;
}

}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();



  return drake::examples::iiwa_soccer::DoMainControlled();
}
