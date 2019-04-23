#  Kuka iiwa plays soccer. Wow!

import trace
import argparse
import numpy as np
import logging
from pydrake.all import (DiagramBuilder, DrakeLcm, SceneGraph,
FindResourceOrThrow, MultibodyPlant,
UniformGravityFieldElement, Simulator, ConnectDrakeVisualizer, Demultiplexer,
Multiplexer, LcmPublisherSystem,
Isometry3, Quaternion, Parser, ConnectSpatialForcesToDrakeVisualizer,
ConnectContactResultsToDrakeVisualizer, ConnectGenericArrowsToDrakeVisualizer)
from box_controller import BoxController

robot_model_name = "box_model"
ball_model_name = "soccer_ball"

ground_model_path = "drake/examples/iiwa_soccer/models/ground.sdf"
arm_model_path = "drake/examples/iiwa_soccer/models/box.sdf"
ball_model_path = "drake/examples/iiwa_soccer/models/soccer_ball.sdf"

def BuildBlockDiagram(mbp_step_size, penetration_allowance, plan_path, fully_actuated):

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
  Parser(plant=robot_plant).AddModelFromFile(file_name=arm_fname)

  # Construct the multibody plant using both the robot and ball models.
  all_plant = mbw_builder.AddSystem(MultibodyPlant(mbp_step_size))
  Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=ground_fname)
  robot_instance_id = Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=arm_fname)
  ball_instance_id = Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=ball_fname)

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

  # The penetration allowance must be set after plant finalization.
  if penetration_allowance >= 0:
      robot_plant.set_penetration_allowance(penetration_allowance)
      all_plant.set_penetration_allowance(penetration_allowance)

  # Connect MBP and SceneGraph.
  mbw_builder.Connect(
          scene_graph.get_query_output_port(),
          all_plant.get_geometry_query_input_port())
  mbw_builder.Connect(
          all_plant.get_geometry_poses_output_port(),
          scene_graph.get_source_pose_port(all_plant.get_source_id()))

  # Export useful ports.
  robot_continuous_state_output = mbw_builder.ExportOutput(all_plant.get_continuous_state_output_port(robot_instance_id))
  ball_continuous_state_output = mbw_builder.ExportOutput(all_plant.get_continuous_state_output_port(ball_instance_id))
  generalized_force_input = mbw_builder.ExportInput(all_plant.get_applied_generalized_force_input_port())

  # Connect Drake Visualizer
  ConnectDrakeVisualizer(builder=mbw_builder, scene_graph=scene_graph)
  ConnectSpatialForcesToDrakeVisualizer(builder=mbw_builder, plant=all_plant)
  ConnectContactResultsToDrakeVisualizer(builder=mbw_builder, plant=all_plant)

  # Add the "MultibodyWorld" to the diagram.
  mbw = builder.AddSystem(mbw_builder.Build())
  mbw.set_name('MultibodyWorld')

  #############################################
  # Add control systems.
  #############################################

  # Build the controller.
  controller = builder.AddSystem(BoxController(mbp_step_size, penetration_allowance, 'box', all_plant, robot_plant, mbw, robot_instance_id, ball_instance_id, fully_actuated))
  controller.LoadPlans(plan_path)
  ConnectGenericArrowsToDrakeVisualizer(builder=builder, output_port=controller.ball_acceleration_visualization_output_port)

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

  # Connect the muxers and controllers to the controller.
  builder.Connect(ball_q_muxer.get_output_port(0), controller.get_input_port_estimated_ball_q())
  builder.Connect(ball_v_muxer.get_output_port(0), controller.get_input_port_estimated_ball_v())
  builder.Connect(robot_q_muxer.get_output_port(0), controller.get_input_port_estimated_robot_q())
  builder.Connect(robot_v_muxer.get_output_port(0), controller.get_input_port_estimated_robot_v())

  # Connect the generalized force input port.
  builder.Connect(controller.get_output_port_control(), mbw.get_input_port(generalized_force_input))

  # Build the diagram.
  diagram = builder.Build()

  return [ controller, diagram, all_plant, robot_plant, mbw, robot_instance, ball_instance, robot_continuous_state_output ]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--fully_actuated", type=str2bool, default=False)
  parser.add_argument(
      "--simulation_time", type=float, default=10.0,
      help="Desired duration of the simulation in seconds.")
  parser.add_argument(
      "--target_realtime_rate", type=float, default=1.0,
      help="Desired rate relative to real time.  See documentation for "
           "Simulator::set_target_realtime_rate() for details.")
  parser.add_argument(
      "--step_size", type=float, default=0.001,
      help="If greater than zero, the plant is modeled as a system with "
           "discrete updates and period equal to this time_step. "
           "If 0, the plant is modeled as a continuous system.")
  parser.add_argument(
      "--penetration_allowance", type=float, default=1e-8,
      help="The amount of interpenetration to allow in the simulation.")
  parser.add_argument(
      "--kp", type=float, default=60.0,
      help="Cartesian Kp for impedance control. Gets used for all xyz "
           "directions.")
  parser.add_argument(
      "--kd", type=float, default=30.0,
      help="Cartesian kd for impedance control. Gets used for all xyz "
           "directions.")
  parser.add_argument(
      "--plan_path", default='plan_box_curve/',
      help='Path to the plan')
  parser.add_argument(
      "--log", default='none',
      help='Logging type: "none", "info", "warning", "debug"')
  args = parser.parse_args()

  # Determine the logging level.
  if args.log.upper() != 'NONE':
      numeric_level = getattr(logging, args.log.upper(), None)
      if not isinstance(numeric_level, int):
          raise ValueError('Invalid log level: %s' % args.log)
      logging.basicConfig(level=numeric_level)
  else:
      logging.disable(logging.CRITICAL)

  controller, diagram, all_plant, robot_plant, mbw, robot_instance, ball_instance, robot_continuous_state_output = BuildBlockDiagram(args.step_size, args.penetration_allowance, args.plan_path, args.fully_actuated)

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
