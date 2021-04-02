#ifndef FIDES_TRUST_REGION_HPP
#define FIDES_TRUST_REGION_HPP

#include "fides/constants.hpp"

#include <blaze/math/Forward.h>

#include <memory>

namespace fides {

using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;

class Step;

/**
 * @brief Compute a step according to the solution of the trust-region
 * subproblem.
 *
 * If step-back is necessary, gradient and reflected trust region step are
 * also evaluated in terms of their performance according to the local
 * quadratic approximation.
 *
 * @param x Current values of the optimization variables
 * @param g Objective function gradient at x
 * @param hess (Approximate) objective function Hessian at x
 * @param scaling Scaling transformation according to distance to boundary
 * @param delta Trust region radius, note that this applies after scaling
 * transformation
 * @param dv Derivative of scaling transformation
 * @param theta Parameter regulating stepback
 * @param lb Lower optimization variable boundaries
 * @param ub Upper optimization variable boundaries
 * @param subspace_dim Subspace dimension in which the subproblem will be
 * solved. Larger subspaces require more compute time but can yield higher
 * quality step proposals.
 * @param stepback_strategy Strategy that is applied when the proposed step
 * exceeds the optimization boundary.
 * @param refine_stepback If set to `true`, proposed steps that are computed
 * via the specified stepback_strategy will be refined via optimization.
 * @return Proposed step
 */
std::unique_ptr<Step> trust_region(
    DynamicVector<double> const &x, DynamicVector<double> const &g,
    DynamicMatrix<double> const &hess, CompressedMatrix<double> const &scaling,
    const double delta, DynamicVector<double> const &dv, const double theta,
    DynamicVector<double> const &lb, DynamicVector<double> const &ub,
    const SubSpaceDim subspace_dim, const StepBackStrategy stepback_strategy,
    const bool refine_stepback);

} // namespace fides

#endif // FIDES_TRUST_REGION_HPP
