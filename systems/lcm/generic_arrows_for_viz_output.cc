#include "drake/lcmt_generic_arrow_for_viz.hpp"
#include "drake/lcmt_generic_arrows_for_viz.hpp"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_output_port.h"
#include "drake/systems/lcm/generic_arrows_for_viz_output.h"

namespace drake {
namespace systems {

namespace internal {

template <class T>
void CalcArrowOutput(
    const Context<T>& context,
    const std::vector<ArrowVisualization>& arrows_to_visualize,
    lcmt_generic_arrows_for_viz* output) {
  auto& msg = *output;

  // Time in microseconds.
  msg.timestamp = static_cast<int64_t>(
      ExtractDoubleOrThrow(context.get_time()) * 1e6);
  msg.num_arrows_to_visualize = static_cast<int>(arrows_to_visualize.size());
  msg.arrows.resize(msg.num_arrows_to_visualize);

  for (int i = 0; i < msg.num_arrows_to_visualize; ++i) {
    lcmt_generic_arrow_for_viz& arrow_msg = msg.arrows[i];
    const ArrowVisualization& arrow_data = arrows_to_visualize[i];
    arrow_msg.timestamp = msg.timestamp;

    auto write_double3 = [](const Vector3<T>& src, double* dest) {
      dest[0] = ExtractDoubleOrThrow(src(0));
      dest[1] = ExtractDoubleOrThrow(src(1));
      dest[2] = ExtractDoubleOrThrow(src(2));
    };
    write_double3(arrow_data.origin_W, arrow_msg.origin_W);
    write_double3(arrow_data.target_W, arrow_msg.target_W);
    write_double3(arrow_data.color_rgb, arrow_msg.color_rgb);
    arrow_msg.fatness_scaling = arrow_data.fatness_scaling;
  }
}
}  // namespace internal

template <class T>
typename LeafOutputPort<T>::CalcCallback CreateArrowOutputCalcCallback(
    std::function<std::vector<ArrowVisualization>(const Context<T>&)>
    arrow_visualization_callback) {
  return [arrow_visualization_callback](
      const Context<T>& context, AbstractValue* output) {
    DRAKE_DEMAND(output);
    const std::vector<ArrowVisualization> arrows_to_visualize =
        arrow_visualization_callback(context);        
    auto& lcm_output = output->GetMutableValue<lcmt_generic_arrows_for_viz>();
    CalcArrowOutput(context, arrows_to_visualize, &lcm_output);
  };
}

systems::lcm::LcmPublisherSystem* ConnectGenericArrowsToDrakeVisualizer(
    systems::DiagramBuilder<double>* builder,
    const systems::OutputPort<double>& arrow_output_port,
    drake::lcm::DrakeLcmInterface* lcm) {
  DRAKE_DEMAND(builder != nullptr);

  auto arrow_publisher = builder->AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_generic_arrows_for_viz>(
          "GENERIC_ARROWS", lcm, 1.0 / 60 /* publish period */));
  arrow_publisher->set_name("arrows_publisher");

  builder->Connect(
      arrow_output_port, arrow_publisher->get_input_port());

  return arrow_publisher;
}

}  // namespace systems
}  // namespace drake

template drake::systems::LeafOutputPort<double>::CalcCallback
    drake::systems::CreateArrowOutputCalcCallback<double>(
        std::function<std::vector<drake::systems::ArrowVisualization>(
            const drake::systems::Context<double>&)>
                arrow_visualization_callback);
