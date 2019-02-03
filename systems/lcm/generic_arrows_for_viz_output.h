#include "drake/systems/framework/leaf_output_port.h"

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

template <class T>
typename LeafOutputPort<T>::CalcCallback CreateArrowOutputLambda(
    std::function<std::vector<ArrowVisualization>(const Context<T>&)>
    arrow_visualization_callback);

}  // namespace systems
}  // namespace drake
