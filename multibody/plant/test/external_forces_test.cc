#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/plant/test/kuka_iiwa_model_tests.h"
#include "drake/multibody/tree/body.h"
#include "drake/multibody/tree/frame.h"

namespace drake {

namespace multibody {
namespace multibody_plant {

using test::KukaIiwaModelTests;

namespace {

TEST_F(KukaIiwaModelTests, ExternalBodyForces) {
  SetArbitraryConfiguration();

  // An arbitrary point on the end effector frame E.
  Vector3<double> p_EP(0.1, -0.05, 0.3);

  // An arbitrary spatial force applied at point P on the end effector,
  // expressed in the end effector frame E.
  const SpatialForce<double> F_Ep_E(
      Vector3<double>(1.0, 2.0, 3.0), Vector3<double>(-5.0, -4.0, -2.0));

  int nv = plant_->num_velocities();

  // Build a vector of generalized accelerations with arbitrary values.
  const VectorX<double> vdot = VectorX<double>::LinSpaced(nv, -5.0, 5.0);

  // Compute inverse dynamics with an externally applied force.
  MultibodyForces<double> forces(*plant_);
  end_effector_link_->AddInForce(
      *context_, p_EP, F_Ep_E, end_effector_link_->body_frame(), &forces);
  const VectorX<double> tau_id =
      plant_->CalcInverseDynamics(*context_, vdot, forces);

  MatrixX<double> M(nv, nv);
  plant_->CalcMassMatrixViaInverseDynamics(*context_, &M);

  VectorX<double> C(nv);
  plant_->CalcBiasTerm(*context_, &C);

  // Frame Jacobian for point p_EP.
  MatrixX<double> Jv_WEp(6, nv);
  plant_->CalcFrameGeometricJacobianExpressedInWorld(
      *context_, end_effector_link_->body_frame(), p_EP, &Jv_WEp);

  // Compute the expected value of inverse dynamics when external forcing is
  // considered.
  const Matrix3<double> R_WE =
      end_effector_link_->EvalPoseInWorld(*context_).linear();
  const SpatialForce<double> F_Ep_W = R_WE * F_Ep_E;
  VectorX<double> tau_id_expected =
      M * vdot + C - Jv_WEp.transpose() * F_Ep_W.get_coeffs();

  // Numerical tolerance used to verify numerical results.
  // Error loss is expected in both forward kinematics and inverse dynamics
  // computations since errors accumulate during inboard/outboard passes.
  const double kTolerance = 50 * std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(CompareMatrices(
      tau_id, tau_id_expected,
      kTolerance, MatrixCompareType::relative));
}

}  // namespace
}  // namespace multibody_plant
}  // namespace multibody
}  // namespace drake
