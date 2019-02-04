#include "drake/lcm/drake_lcm_interface.h"
#include "drake/lcmt_generic_arrows_for_viz.hpp"
#include "drake/systems/framework/leaf_output_port.h"
#include "drake/systems/lcm/lcm_publisher_system.h"

namespace drake {
namespace systems {

struct ArrowVisualization {
  // Point of origin of the arrow in the world frame.
  Vector3<double> origin_W;   

  // Target point of the arrow in the world frame.
  Vector3<double> target_W;

  // Color of the arrow.
  Vector3<double> color_rgb;

  // Fatness scaling of the arrow.
  double fatness_scaling{1.0};
};

systems::lcm::LcmPublisherSystem* ConnectGenericArrowsToDrakeVisualizer(
    systems::DiagramBuilder<double>* builder,
    const systems::OutputPort<double>& arrow_output_port,
    drake::lcm::DrakeLcmInterface* lcm = nullptr);

template <class T>
typename LeafOutputPort<T>::AllocCallback CreateArrowOutputAllocCallback() {
  return []() -> std::unique_ptr<AbstractValue> {
    return std::make_unique<Value<lcmt_generic_arrows_for_viz>>();
  };
}

template <class T>
typename LeafOutputPort<T>::CalcCallback CreateArrowOutputCalcCallback(
    std::function<std::vector<ArrowVisualization>(const Context<T>&)>
    arrow_visualization_callback);

}  // namespace systems
}  // namespace drake
