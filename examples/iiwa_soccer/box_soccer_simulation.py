#  Kuka iiwa plays soccer. Wow!

import trace
import argparse
import numpy as np
from pydrake.all import (DiagramBuilder, DrakeLcm, SceneGraph,
FindResourceOrThrow, MultibodyPlant, AddModelFromSdfFile,
UniformGravityFieldElement, Simulator, ConnectDrakeVisualizer, Demultiplexer,
Multiplexer, LcmPublisherSystem, MobilizerIndex, ConstantVectorSource,
Isometry3, Quaternion, Parser, ConnectSpatialForcesToDrakeVisualizer,
ConnectContactResultsToDrakeVisualizer)
from box_controller import BoxController

robot_model_name = "box_model"
ball_model_name = "soccer_ball"

ground_model_path = "drake/examples/iiwa_soccer/models/ground.sdf"
arm_model_path = "drake/examples/iiwa_soccer/models/box.sdf"
ball_model_path = "drake/examples/iiwa_soccer/models/soccer_ball.sdf"

def BuildBlockDiagram(mbp_step_size, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd):

  # Construct DiagramBuilder objects for both "MultibodyWorld" and the total
  # diagram (comprising all systems).
  builder = DiagramBuilder()
  mbw_builder = DiagramBuilder()
  scene_graph = mbw_builder.AddSystem(SceneGraph())

  # Get the model paths.
  ground_fname = FindResourceOrThrow(ground_model_path)
  arm_fname = FindResourceOrThrow(arm_model_path)
  ball_fname = FindResourceOrThrow(ball_model_path)

  # Construct a multibody plant just for kinematics/dynamics calculations.
  robot_plant = MultibodyPlant(mbp_step_size)
  AddModelFromSdfFile(file_name=arm_fname, plant=robot_plant)
  #Parser(plant=robot_plant).AddModelFromFile(file_name=arm_fname)

  # Construct the multibody plant using both the robot and ball models.
  all_plant = mbw_builder.AddSystem(MultibodyPlant(mbp_step_size))
  robot_instance_id = AddModelFromSdfFile(file_name=arm_fname, plant=all_plant,
                                          scene_graph=scene_graph)
  ball_instance_id = AddModelFromSdfFile(file_name=ball_fname, plant=all_plant,
                                         scene_graph=scene_graph)
  AddModelFromSdfFile(file_name=ground_fname, plant=all_plant, scene_graph=scene_graph)
  #robot_instance_id = Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=arm_fname)
  #ball_instance_id = Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=ball_fname)

  # Weld the ground to the world.
  all_plant.WeldFrames(all_plant.world_frame(), all_plant.GetFrameByName("ground_body"))

  # Add gravity to the models.
  all_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))
  robot_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))

  # Finalize the plants.
  robot_plant.Finalize()
  all_plant.Finalize(scene_graph)
  assert robot_plant.num_actuators() == 0
  assert all_plant.num_actuators() == 0
  assert all_plant.geometry_source_is_registered()

  # Connect MBP and SceneGraph.
  mbw_builder.Connect(
    scene_graph.get_query_output_port(),
    all_plant.get_geometry_query_input_port())
  mbw_builder.Connect(
    all_plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(all_plant.get_source_id()))

  # Connect Drake Visualizer
  ConnectDrakeVisualizer(builder=mbw_builder, scene_graph=scene_graph)
  ConnectSpatialForcesToDrakeVisualizer(builder=mbw_builder, plant=all_plant)
  ConnectContactResultsToDrakeVisualizer(builder=mbw_builder, plant=all_plant)

  # Export useful ports.
  robot_continuous_state_output = mbw_builder.ExportOutput(all_plant.get_continuous_state_output_port(robot_instance_id))
  ball_continuous_state_output = mbw_builder.ExportOutput(all_plant.get_continuous_state_output_port(ball_instance_id))
  robot_god_input = mbw_builder.ExportInput(all_plant.get_god_input_port(robot_instance_id))
  ball_god_input = mbw_builder.ExportInput(all_plant.get_god_input_port(ball_instance_id))

  # Add the "MultibodyWorld" to the diagram.
  mbw = builder.AddSystem(mbw_builder.Build())
  mbw.set_name('MultibodyWorld')

  #############################################
  # Add control systems.
  #############################################

  # Build the controller.
  controller = builder.AddSystem(BoxController('box', all_plant, robot_plant, mbw, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd, robot_instance_id, ball_instance_id))

  # Get the necessary instances.
  robot_instance = all_plant.GetModelInstanceByName(robot_model_name)
  ball_instance = all_plant.GetModelInstanceByName(ball_model_name)

  # Construct the necessary demultiplexers.
  nq_ball = all_plant.num_positions(ball_instance)
  nq_robot = all_plant.num_positions(robot_instance)
  nv_ball = all_plant.num_velocities(ball_instance)
  nv_robot = all_plant.num_velocities(robot_instance)
  robot_state_demuxer = builder.AddSystem(Demultiplexer(
    nq_robot + nv_robot, 1))
  ball_state_demuxer = builder.AddSystem(Demultiplexer(
    nq_ball + nv_ball, 1))

  # Construct the necessary multiplexers for the robot state.
  # Note: these must be changed if robot DOF changes.
  robot_config_port_input_sizes = [1] * nq_robot
  robot_vel_port_input_sizes = [1] * nv_robot
  robot_q_muxer = builder.AddSystem(Multiplexer(robot_config_port_input_sizes))
  robot_v_muxer = builder.AddSystem(Multiplexer(robot_vel_port_input_sizes))

  # Construct the necessary mutliplexers for the ball state.
  ball_config_port_input_sizes = [1] * nq_ball
  ball_vel_port_input_sizes = [1] * nv_ball
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

  # Connect the muxers and controllers to the MBW.
  builder.Connect(ball_q_muxer.get_output_port(0), controller.get_input_port_estimated_ball_q())
  builder.Connect(ball_v_muxer.get_output_port(0), controller.get_input_port_estimated_ball_v())
  builder.Connect(robot_q_muxer.get_output_port(0), controller.get_input_port_estimated_robot_q())
  builder.Connect(robot_v_muxer.get_output_port(0), controller.get_input_port_estimated_robot_v())
  builder.Connect(controller.get_output_port_control(), mbw.get_input_port(robot_god_input))

  # Construct a constant source to "plug" the ball God input.
  zero_source = builder.AddSystem(ConstantVectorSource(np.zeros([controller.nv_ball()])))
  builder.Connect(zero_source.get_output_port(0), mbw.get_input_port(ball_god_input))

  # Build the diagram.
  diagram = builder.Build()

  return [ controller, diagram, all_plant, robot_plant, mbw, robot_instance, ball_instance ]

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--simulation_time", type=float, default=10.0,
      help="Desired duration of the simulation in seconds.")
  parser.add_argument(
      "--target_realtime_rate", type=float, default=1.0,
      help="Desired rate relative to real time.  See documentation for "
           "Simulator::set_target_realtime_rate() for details.")
  parser.add_argument(        "--time_step", type=float, default=0.01,
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

  # Gains in Cartesian-land.
  robot_cart_kp = np.ones([3, 1]) * args.kp
  robot_cart_kd = np.ones([3, 1]) * args.kd

  # Joint gains for the robot.
  nv_robot = 6
  robot_gv_kp = np.ones([nv_robot]) * 10
  robot_gv_ki = np.ones([nv_robot]) * 0.1
  robot_gv_kd = np.ones([nv_robot]) * 1.0

  controller, diagram, all_plant, robot_plant, mbw, robot_instance, ball_instance = BuildBlockDiagram(args.time_step, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd)

  simulator = Simulator(diagram)
  simulator.set_publish_every_time_step(True)
  simulator.set_target_realtime_rate(args.target_realtime_rate)

  # Set the initial conditions, according to the plan.
  context = simulator.get_mutable_context()

  # Set the robot and ball state to the one at the specified time in the plan.
  mbw_context = diagram.GetMutableSubsystemContext(mbw, context)
  robot_and_ball_context = mbw.GetMutableSubsystemContext(all_plant, mbw_context)
  plan = controller.plan
  x_robot = plan.GetRobotQVAndVdot(0)[0:controller.nq_robot()+controller.nv_robot()]
  x_ball = plan.GetBallQVAndVdot(0)[0:controller.nq_ball()+controller.nv_ball()]
  all_plant.SetPositionsAndVelocities(robot_and_ball_context, robot_instance, x_robot)
  all_plant.SetPositionsAndVelocities(robot_and_ball_context, ball_instance, x_ball)

  # Step to the time.
  simulator.Initialize()
  simulator.StepTo(args.simulation_time)

if __name__ == "__main__":
  #tracer = trace.Trace(trace=1, count=0)
  #tracer.run('main()')
  main()
