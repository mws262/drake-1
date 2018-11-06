#  Kuka iiwa plays soccer. Wow!

import argparse
import numpy as np
from pydrake.all import (DiagramBuilder, DrakeLcm, SceneGraph,
FindResourceOrThrow, MultibodyPlant, AddModelFromSdfFile,
UniformGravityFieldElement, Simulator, ConnectDrakeVisualizer, Demultiplexer,
Multiplexer, LcmPublisherSystem, MobilizerIndex)
from box_controller import BoxController

arm_model_path = "drake/examples/iiwa_soccer/models/box.sdf"
ball_model_path = "drake/examples/iiwa_soccer/models/soccer_ball.sdf"

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--simulation_time", type=float, default=10.0,
      help="Desired duration of the simulation in seconds.")
  parser.add_argument(
      "--target_realtime_rate", type=float, default=1.0,
      help="Desired rate relative to real time.  See documentation for "
           "Simulator::set_target_realtime_rate() for details.")
  parser.add_argument(        "--time_step", type=float, default=0.,
      help="If greater than zero, the plant is modeled as a system with "
           "discrete updates and period equal to this time_step. "
           "If 0, the plant is modeled as a continuous system.")
  parser.add_argument(
      "--kp", type=float, default=60.0,
      help="Cartesian Kp for impedance control. Gets used for all xyz "
           "directions.")
  parser.add_argument(
      "--kd", type=float, default=30.0,
      help="Cartesian kd for impedance control. Gets used for all xyz "
           "directions.")
  args = parser.parse_args()

  # Set up the world.
  lcm = DrakeLcm()  # Communications to robot or visualizer.

  # Construct DiagramBuilder objects for both "MultibodyWorld" and the total
  # diagram (comprising all systems).
  builder = DiagramBuilder()
  mbw_builder = DiagramBuilder()
  scene_graph = mbw_builder.AddSystem(SceneGraph())

  # Get the model paths.
  arm_fname = FindResourceOrThrow(arm_model_path)
  ball_fname = FindResourceOrThrow(ball_model_path)

  # Construct a multibody plant just for kinematics/dynamics calculations.
  robot_plant = builder.AddSystem(MultibodyPlant(time_step=args.time_step))
  AddModelFromSdfFile(file_name=arm_fname, plant=robot_plant)

  # Construct the multibody plant using both the robot and ball models.
  all_plant = mbw_builder.AddSystem(MultibodyPlant(time_step=args.time_step))
  robot_instance_id = AddModelFromSdfFile(file_name=arm_fname, plant=all_plant,
                      scene_graph=scene_graph)
  ball_instance_id = AddModelFromSdfFile(file_name=ball_fname, plant=all_plant,
                      scene_graph=scene_graph)

  # Add gravity to the models.
  all_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))
  robot_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))

  # Finalize the plants.
  robot_plant.Finalize()
  all_plant.Finalize(scene_graph)
  assert all_plant.geometry_source_is_registered()

  # Connect MBP and SceneGraph.
  mbw_builder.Connect(
    scene_graph.get_query_output_port(),
    all_plant.get_geometry_query_input_port())
  mbw_builder.Connect(
    all_plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(all_plant.get_source_id()))

  # Connect Drake Visualizer
  ConnectDrakeVisualizer(builder=mbw_builder, scene_graph=scene_graph, lcm=lcm)

  # Export useful ports.
  robot_continuous_state_output = mbw_builder.ExportOutput(all_plant.get_continuous_state_output_port(robot_instance_id))
  ball_continuous_state_output = mbw_builder.ExportOutput(all_plant.get_continuous_state_output_port(ball_instance_id))
  robot_actuation_input = mbw_builder.ExportInput(all_plant.get_actuation_input_port(robot_instance_id))

  # Add the "MultibodyWorld" to the diagram.
  mbw = builder.AddSystem(mbw_builder.Build())

  # Create a context for MBW.
  mbw_context = mbw.CreateDefaultContext()

  # Add control systems.

  # Gains in Cartesian-land.
  k_p = np.ones([3, 1]) * args.kp
  k_d = np.ones([3, 1]) * args.kd

  # Joint gains for the robot.
  nv_robot = 6
  joint_kp = np.ones([nv_robot, 1]) * 10
  joint_ki = np.ones([nv_robot, 1]) * 0.1
  joint_kd = np.ones([nv_robot, 1]) * 1.0

  controller = builder.AddSystem(BoxController(all_plant, robot_plant, mbw, k_p, k_d, joint_kp, joint_ki, joint_kd, robot_instance_id, ball_instance_id))

  # TODO: Make this more robust.
  # Construct the necessary demultiplexers.
  nq_ball = 7
  nq_robot = 6
  nv_ball = 6
  nv_robot = 6
  robot_state_demuxer = builder.AddSystem(Demultiplexer(
      nq_robot + nv_robot, 1))
  ball_state_demuxer = builder.AddSystem(Demultiplexer(
      nq_ball + nv_ball, 1))

  # Construct the necessary multiplexers for the robot state.
  # Note: these must be changed if robot DOF changes.
  robot_config_port_input_sizes = [ 1, 1, 1, 1, 1, 1 ]
  robot_vel_port_input_sizes = [ 1, 1, 1, 1, 1, 1 ]
  robot_q_muxer = builder.AddSystem(Multiplexer(robot_config_port_input_sizes))
  robot_v_muxer = builder.AddSystem(Multiplexer(robot_vel_port_input_sizes))

  # Construct the necessary mutliplexers for the ball state.
  ball_config_port_input_sizes = [ 1, 1, 1, 1, 1, 1, 1 ]
  ball_vel_port_input_sizes = [ 1, 1, 1, 1, 1, 1 ]
  ball_q_muxer = builder.AddSystem(Multiplexer(ball_config_port_input_sizes))
  ball_v_muxer = builder.AddSystem(Multiplexer(ball_vel_port_input_sizes))

  # Connect the demultiplexers and multiplexers.
  builder.Connect(mbw.get_output_port(robot_continuous_state_output), robot_state_demuxer.get_input_port(0))
  builder.Connect(mbw.get_output_port(ball_continuous_state_output), ball_state_demuxer.get_input_port(0))
  for i in range(nq_ball):
    builder.Connect(ball_state_demuxer.get_output_port(i), ball_q_muxer.get_input_port(i))
  for i in range(nq_robot):
      builder.Connect(robot_state_demuxer.get_output_port(i), robot_q_muxer.get_input_port(i))
  for i in range(nv_ball):
    builder.Connect(ball_state_demuxer.get_output_port(nq_ball + i), ball_v_muxer.get_input_port(i))
  for i in range(nv_robot):
      builder.Connect(robot_state_demuxer.get_output_port(nq_robot + i), robot_v_muxer.get_input_port(i))
  print 'Connecting 1'
  builder.Connect(ball_q_muxer.get_output_port(0), controller.get_input_port_estimated_ball_q())
  print 'Connecting 2'
  builder.Connect(ball_v_muxer.get_output_port(0), controller.get_input_port_estimated_ball_v())
  print 'Connecting 3'
  builder.Connect(robot_state_demuxer.get_output_port(0), controller.get_input_port_estimated_robot_q())
  print 'Connecting 4'
  builder.Connect(robot_state_demuxer.get_output_port(1), controller.get_input_port_estimated_robot_qd())
  print 'Connecting 5'
  builder.Connect(controller.get_output_port_control(), all_plant.get_actuation_input_port())
  print 'Connecting 6'

  # Build the diagram.
  diagram = builder.Build()

  '''
  #####################
  std::vector<Eigen::Isometry3d> poses_to_draw
  Eigen::Isometry3d target_pose
  target_pose.setIdentity()
  Eigen::Vector3d tarPos

  #####################
  '''
  simulator = Simulator(diagram)
  simulator.set_publish_every_time_step(True)
  simulator.set_target_realtime_rate(args.target_realtime_rate)

  # Set the initial conditions, according to the plan.
  context = simulator.get_mutable_context()

  '''
  plan = controller.plan()
  t0 = 0
  q = np.zeros([nq_ball + nq_robot, 1])
  v = np.zeros((nv_ball + nv_robot, 1])
  q_robot = plan.GetRobotQQdotAndQddot(t0)[0:nq_robot-1]
  v_robot = plan.GetRobotQQdotAndQddot(t0)[nq_robot:nq_robot+nv_robot-1]
  q_ball = plan.GetBallQVAndVdot(t0)[0:nq_ball-1]
  v_ball = plan.GetBallQVAndVdot(t0)[nq_ball:nq_ball+nv_ball-1]

  # Set q and v for the robot and the ball.
  all_tree = all_plant.tree()
  robot_instance = all_tree.GetModelInstanceByName(robot_model_name)
  ball_instance = all_tree.GetModelInstanceByName(ball_model_name)
  x = np.zeros([nq_robot + nv_robot + nq_ball + nv_ball, 1])
  all_tree.set_positions_in_array(robot_instance, q_robot, &q)
  all_tree.set_positions_in_array(ball_instance, q_ball, &q)
  all_tree.set_velocities_in_array(robot_instance, v_robot, &v)
  all_tree.set_velocities_in_array(ball_instance, v_ball, &v)
  x[0:len(q)-1] = q
  x[-len(v):] = v
  all_plant.tree().get_mutable_multibody_state_vector(context) = x
  ''' 
  simulator.Initialize()
#  simulator.StepTo(args.simulation_time)
  simulator.StepTo(.1)

main()

