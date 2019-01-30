#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/multibody/tree/multibody_tree.h"

namespace drake {
namespace multibody {

template <class T>
struct SpatialForceOutput {
  SpatialForceOutput(
    const Vector3<T>& point_W, const SpatialForce<T>& Force_p_W) :
      p_W(point_W), F_p_W(Force_p_W) { }

  /// Point of application of the spatial force, where the point represents
  /// a vector expressed in the world frame.
  Vector3<T> p_W;

  /// Spatial force applied at point p and expressed in the world frame.
  SpatialForce<T> F_p_W;
};

}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class drake::multibody::SpatialForceOutput)
