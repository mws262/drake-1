#  Kuka iiwa plays soccer. Wow!

import trace
import argparse
import numpy as np
from pydrake.all import (DiagramBuilder, DrakeLcm, SceneGraph,
FindResourceOrThrow, MultibodyPlant, AddModelFromSdfFile,
UniformGravityFieldElement, Simulator, ConnectDrakeVisualizer, Demultiplexer,
Multiplexer, LcmPublisherSystem, MobilizerIndex, ConstantVectorSource,
Isometry3, Quaternion, Parser)

robot_model_name = "box_model"
ball_model_name = "soccer_ball"

ground_model_path = "drake/examples/iiwa_soccer/models/ground.sdf"
arm_model_path = "drake/examples/iiwa_soccer/models/box.sdf"
ball_model_path = "drake/examples/iiwa_soccer/models/soccer_ball.sdf"

class EmbeddedSim:

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

    # Construct the multibody plant using both the robot and ball models.
    all_plant = mbw_builder.AddSystem(MultibodyPlant(mbp_step_size))
    robot_instance_id = AddModelFromSdfFile(file_name=arm_fname, plant=all_plant,
                                            scene_graph=scene_graph)
    ball_instance_id = AddModelFromSdfFile(file_name=ball_fname, plant=all_plant,
                                          scene_graph=scene_graph)
    AddModelFromSdfFile(file_name=ground_fname, plant=all_plant, scene_graph=scene_graph)

    # Weld the ground to the world.
    all_plant.WeldFrames(all_plant.world_frame(), all_plant.GetFrameByName("ground_body"))

    # Add gravity to the models.
    all_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))
    robot_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))

    # Finalize the plants.
    all_plant.Finalize(scene_graph)
    assert all_plant.num_actuators() == 0
    assert all_plant.geometry_source_is_registered()

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
    robot_god_input = mbw_builder.ExportInput(all_plant.get_god_input_port(robot_instance_id))
    ball_god_input = mbw_builder.ExportInput(all_plant.get_god_input_port(ball_instance_id))

    # Add the "MultibodyWorld" to the diagram.
    mbw = builder.AddSystem(mbw_builder.Build())

    #############################################
    # Add control systems.
    #############################################

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

    # Build the controller.
    control_input = builder.AddSystem(ConstantVectorSource(np.zeros([nv_robot])))

    # Connect the controller to the MBW.
    builder.Connect(control_input.get_output_port(0), mbw.get_input_port(robot_god_input))

    # Construct a constant source to "plug" the ball God input.
    zero_source = builder.AddSystem(ConstantVectorSource(np.zeros([controller.nv_ball()])))
    builder.Connect(zero_source.get_output_port(0), mbw.get_input_port(ball_god_input))

    # Build the diagram.
    diagram = builder.Build()

    return [ control_input, diagram, all_plant, mbw, robot_instance, ball_instance ]

  def ApplyControls(self, u):
    control_input.get_mutable_source_value().SetFromVector(u)

  def StepEmbeddedSimulation(self, dt):


  def ConstructEmbeddedSimulation(self, dt):

    self.delta_t = dt
    self.control_input, self.diagram, self.all_plant, mbw, robot_instance, ball_instance = BuildBlockDiagram(dt, robot_cart_kp, robot_cart_kd, robot_gv_kp, robot_gv_ki, robot_gv_kd)

    self.simulator = Simulator(diagram)
    self.simulator.set_publish_every_time_step(True)
    self.simulator.set_target_realtime_rate(args.target_realtime_rate)

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
