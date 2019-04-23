#  Kuka iiwa plays soccer. Wow!

import trace
import argparse
import numpy as np
from pydrake.all import (DiagramBuilder, SceneGraph, FindResourceOrThrow, MultibodyPlant, UniformGravityFieldElement, \
    Simulator, ConstantVectorSource, Isometry3, Quaternion, Parser, Demultiplexer)

robot_model_name = "box_model"
ball_model_name = "soccer_ball"

ground_model_path = "drake/examples/iiwa_soccer/models/ground.sdf"
arm_model_path = "drake/examples/iiwa_soccer/models/box.sdf"
ball_model_path = "drake/examples/iiwa_soccer/models/soccer_ball.sdf"

class EmbeddedSim:

  def __init__(self, dt, penetration_allowance):
      self.delta_t = dt
      self.control_input, self.diagram, self.all_plant, self.mbw, robot_instance, ball_instance = self.BuildBlockDiagram(dt, penetration_allowance)

      self.simulator = Simulator(self.diagram)
      self.simulator.set_publish_every_time_step(True)

      # Set the initial conditions, according to the plan.
      self.context = self.simulator.get_mutable_context()

  # Constructs the necessary block diagram.
  def BuildBlockDiagram(self, mbp_step_size, penetration_allowance):
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
    self.robot_and_ball_plant = all_plant
    Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=ground_fname)
    robot_instance_id = Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=arm_fname)
    ball_instance_id = Parser(plant=all_plant, scene_graph=scene_graph).AddModelFromFile(file_name=ball_fname)

    # Weld the ground to the world.
    all_plant.WeldFrames(all_plant.world_frame(), all_plant.GetFrameByName("ground_body"))

    # Add gravity to the models.
    all_plant.AddForceElement(UniformGravityFieldElement([0, 0, -9.81]))

    # Finalize the plants.
    all_plant.Finalize(scene_graph)
    if penetration_allowance >= 0:
        all_plant.set_penetration_allowance(penetration_allowance)
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
    robot_continuous_state_output = mbw_builder.ExportOutput(
        all_plant.get_continuous_state_output_port(robot_instance_id))
    ball_continuous_state_output = mbw_builder.ExportOutput(
        all_plant.get_continuous_state_output_port(ball_instance_id))
    generalized_force_input = mbw_builder.ExportInput(all_plant.get_applied_generalized_force_input_port())

    # Add the "MultibodyWorld" to the diagram.
    mbw = builder.AddSystem(mbw_builder.Build())

    #############################################
    # Add control systems.
    #############################################

    # Get the necessary instances.
    self.robot_instance = all_plant.GetModelInstanceByName(robot_model_name)
    self.ball_instance = all_plant.GetModelInstanceByName(ball_model_name)

    # Get necessary dimensions.
    nq_ball = all_plant.num_positions(self.ball_instance)
    nq_robot = all_plant.num_positions(self.robot_instance)
    nv_ball = all_plant.num_velocities(self.ball_instance)
    nv_robot = all_plant.num_velocities(self.robot_instance)

    # Build the controller.
    control_input = builder.AddSystem(ConstantVectorSource(np.zeros([nv_robot + nv_ball])))
    builder.Connect(control_input.get_output_port(0), mbw.get_input_port(generalized_force_input))

    # Build the diagram.
    diagram = builder.Build()

    return [ control_input, diagram, all_plant, mbw, self.robot_instance, self.ball_instance ]

  def ApplyControls(self, Bu):
      control_context = self.diagram.GetMutableSubsystemContext(self.control_input, self.context)
      self.control_input.get_mutable_source_value(control_context).SetFromVector(Bu)

  def UpdateTime(self, t):
      self.context.set_time(t)

  def GetPlantVelocities(self):
      mbw_context = self.diagram.GetMutableSubsystemContext(self.mbw, self.context)
      robot_and_ball_context = self.mbw.GetMutableSubsystemContext(self.all_plant, mbw_context)
      return self.all_plant.GetVelocities(robot_and_ball_context)

  def UpdatePlantPositions(self, q):
      mbw_context = self.diagram.GetMutableSubsystemContext(self.mbw, self.context)
      robot_and_ball_context = self.mbw.GetMutableSubsystemContext(self.all_plant, mbw_context)
      self.all_plant.SetPositions(robot_and_ball_context, q)

  def UpdatePlantVelocities(self, v):
      mbw_context = self.diagram.GetMutableSubsystemContext(self.mbw, self.context)
      robot_and_ball_context = self.mbw.GetMutableSubsystemContext(self.all_plant, mbw_context)
      self.all_plant.SetVelocities(robot_and_ball_context, v)

  def CalcAccelerations(self):
      mbw_context = self.diagram.GetMutableSubsystemContext(self.mbw, self.context)
      robot_and_ball_context = self.mbw.GetMutableSubsystemContext(self.all_plant, mbw_context)
      return self.all_plant.CalcAccelerations(robot_and_ball_context, self.delta_t)

  def Step(self):
      self.simulator.Initialize()
      self.simulator.StepTo(self.context.get_time() + self.delta_t)


