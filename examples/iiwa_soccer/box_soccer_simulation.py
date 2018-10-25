#  Kuka iiwa plays soccer. Wow!

import argparse
from pydrake.all import (DiagramBuilder)

#DEFINE_double(kp, 60, "Cartesian kp for impedance control. Gets used for all xyz directions.");
#DEFINE_double(kd, 30, "Cartesian kp for impedance control. Gets used for all xyz directions.");

arm_model_path = "drake/examples/iiwa_soccer/models/box.sdf"
ball_model_path = "drake/examples/iiwa_soccer/models/soccer_ball.sdf"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation_time", type=float, default=10.0,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(        "--time_step", type=float, default=0.,
        help="If greater than zero, the plant is modeled as a system with "
             "discrete updates and period equal to this time_step. "
             "If 0, the plant is modeled as a continuous system.")
    args = parser.parse_args()


  # Set up the world.
  drake::lcm::DrakeLcm lcm; # Communications to robot or visualizer.

  mbw_builder = DiagramBuilder();
  scene_graph = mbw_builder.AddSystem(SceneGraph())

  # Get the model paths.
  arm_fname = FindResourceOrThrow(arm_model_path)
  ball_fname = FindResourceOrThrow(ball_model_path)

  # Construct the multibody plant using both the robot and ball models.
  all_plant = mbw_builder.AddSystem(MultibodyPlant(time_step=args.time_step))
  AddModelFromSdfFile(file_name=arm_fname, plant=all_plant,
                      scene_graph=scene_graph)
  AddModelFromSdfFile(file_name=ball_fname, plant=all_plant,
                      scene_graph=scene_graph)

  # Add gravity to the model.
  all_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))
  all_plant.Finalize(scene_graph)
  assert all_plant.geometry_source_is_registered()

  # Connect MBP and SceneGraph.
  mbw_builder.Connect(
    scene_graph.get_query_output_port(),
    all_plant.get_geometry_query_input_port())
  mbw_builder.Connect(
    all_plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(mbw_builder.get_source_id()))
  mbw = mbw_builder.Build()

  # Create a context for MBW.
  mbw_context = mbw.CreateDefaultContext()

  # Connect Drake Visualizer

  # Add systems.

  # Gains in cartesian-land.
  Vector3<double> k_p;
  k_p << FLAGS_kp, FLAGS_kp, FLAGS_kp;
  Vector3<double> k_d;
  k_d << FLAGS_kd, FLAGS_kd, FLAGS_kd;

  # Joint gains for the robot.
  VectorXd joint_kp(7), joint_ki(7), joint_kd(7);
  joint_kp.setOnes() *= 10.0;
  joint_ki.setOnes() *= 0.1;
  joint_kd.setOnes() *= 1.0;

  controller = builder.AddSystem<BoxController>(
      all_plant, robot_plant, k_p, k_d,
      joint_kp, joint_ki, joint_kd, lcm);

  # TODO: Make this more robust.
  # Construct the necessary demultiplexers.
  const int nq_ball = 7, nq_robot = 7, nv_ball = 6, nv_robot = 7;
  auto robot_state_demuxer = builder.AddSystem<Demultiplexer<double>>(
      nq_robot + nv_robot, nq_robot);
  auto ball_state_demuxer = builder.AddSystem<Demultiplexer<double>>(
      nq_ball + nv_ball, 1);

  # TODO: Make this more robust.
  # Construct the necessary mutliplexers.
  std::vector<int> ball_config_port_input_sizes = { 1, 1, 1, 1,  1, 1, 1 };
  std::vector<int> ball_vel_port_input_sizes = { 1, 1, 1, 1, 1, 1 };
  auto ball_q_muxer = builder.AddSystem<Multiplexer<double>>(
      ball_config_port_input_sizes);
  auto ball_v_muxer = builder.AddSystem<Multiplexer<double>>(
      ball_vel_port_input_sizes);

  # Constant source 2.
  # FULL IIWA INFO TO LCM

  iiwa_status_pub = builder.AddSystem(systems::lcm::LcmPublisherSystem::Make<lcmt_iiwa_status>("IIWA_STATUS", &lcm));
  iiwa_status_pub->set_name("iiwa_status_publisher");
  iiwa_status_pub->set_publish_period(kIiwaLcmStatusPeriod);
  iiwa_status_sender = builder.AddSystem<IiwaStatusSender>(7);
  iiwa_status_sender->set_name("iiwa_status_sender");
  iiwa_command_receiver = builder.AddSystem<IiwaCommandReceiver>(7);
  iiwa_command_receiver->set_name("iwwa_command_receiver");
  iiwa_command_sub = builder.AddSystem(systems::lcm::LcmSubscriberSystem::Make<lcmt_iiwa_command>("IIWA_COMMAND", &lcm));
  iiwa_command_sub->set_name("iiwa_command_subscriber");

  builder.Connect(iiwa_command_sub->get_output_port(), iiwa_command_receiver->get_input_port(0));
  builder.Connect(all_plant.get_continuous_state_output_port(robot_id_table), iiwa_status_sender->get_state_input_port());
  builder.Connect(iiwa_command_receiver->get_output_port(0), iiwa_status_sender->get_command_input_port());
  builder.Connect(all_plant.get_continuous_state_output_port(robot_id_table), robot_state_demuxer->get_input_port(0));
  builder.Connect(all_plant.get_continuous_state_output_port(ball_id_table), ball_state_demuxer->get_input_port(0));
  for (int i = 0; i < nq_ball; ++i)
    builder.Connect(ball_state_demuxer->get_output_port(i), ball_q_muxer->get_input_port(i));
  for (int i = 0; i < nv_ball; ++i)
    builder.Connect(ball_state_demuxer->get_output_port(nq_ball + i), ball_v_muxer->get_input_port(i));
  builder.Connect(ball_q_muxer->get_output_port(0), controller->get_input_port_estimated_ball_q());
  builder.Connect(ball_v_muxer->get_output_port(0), controller->get_input_port_estimated_ball_v());
  builder.Connect(robot_state_demuxer->get_output_port(0), controller->get_input_port_estimated_robot_q());
  builder.Connect(robot_state_demuxer->get_output_port(1), controller->get_input_port_estimated_robot_qd());
  builder.Connect(controller->get_output_port_control(), iiwa_status_sender->get_commanded_torque_input_port());
  builder.Connect(controller->get_output_port_control(), all_plant.get_actuation_input_port());
  builder.Connect(iiwa_status_sender->get_output_port(0), iiwa_status_pub->get_input_port());

  # Last thing before building the diagram; configure the system for
  # visualization.
  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);
  diagram = builder.Build();

  #####################
  std::vector<Eigen::Isometry3d> poses_to_draw;
  Eigen::Isometry3d target_pose;
  target_pose.setIdentity();
  Eigen::Vector3d tarPos;

  #####################

  Simulator<double> simulator(*diagram);
  simulator.set_publish_at_initialization(false);
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);

  # Set the initial conditions, according to the plan.
  Context<double>& context = simulator.get_mutable_context();
  const ManipulationPlan& plan = controller->plan();
  const double t0 = 0;
  VectorXd q(nq_ball + nq_robot);
  VectorXd v(nv_ball + nv_robot);
  q_robot = plan.GetRobotQQdotAndQddot(t0).head(nq_robot);
  v_robot = plan.GetRobotQQdotAndQddot(t0).segment(
      nq_robot, nv_robot);
  q_ball = plan.GetBallQVAndVdot(t0).head(nq_ball);
  v_ball = plan.GetBallQVAndVdot(t0).segment(nq_ball, nv_ball);

  # Set q and v for the robot and the ball.
  all_tree = all_plant.tree();
  robot_instance = all_tree.GetModelInstanceByName(
      std::string(robot_model_name));
  ball_instance = all_tree.GetModelInstanceByName(
      std::string(ball_model_name));
  x(nq_robot + nv_robot + nq_ball + nv_ball);
  all_tree.set_positions_in_array(robot_instance, q_robot, &q);
  all_tree.set_positions_in_array(ball_instance, q_ball, &q);
  all_tree.set_velocities_in_array(robot_instance, v_robot, &v);
  all_tree.set_velocities_in_array(ball_instance, v_ball, &v);
  x.head(q.size()) = q;
  x.tail(v.size()) = v;
  all_plant.tree().get_mutable_multibody_state_vector(&context) = x;
  
  simulator.reset_integrator<systems::RungeKutta2Integrator<double>>(
      *diagram, 1e-3, &context);
  simulator.Initialize();
  simulator.StepTo(FLAGS_simulation_sec);
