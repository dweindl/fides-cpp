/**
 * Trust Region StepBack
 * -----------------------------
 * This module provides the machinery to combine various step-back strategies
 * that can be used to compute longer steps in case the initially proposed step
 * had to be truncated due to non-compliance with boundary constraints.
 */

#ifndef FIDES_STEPBACK_HPP
#define FIDES_STEPBACK_HPP

#include <blaze/Forward.h>

#include "fides/steps.hpp"

namespace fides {

using ::blaze::CompressedMatrix;
using ::blaze::DynamicMatrix;
using ::blaze::DynamicVector;

/**
 * @brief Compute new proposal steps according to a reflection strategy.
 * @param tr_step Reference trust region step that will be reflect
 * @param x Current values of the optimization variables
 * @param sg Rescaled objective function gradient at x
 * @param hess (Approximate) objective function Hessian at x
 * @param scaling Scaling transformation according to distance to boundary
 * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
 * transformation
 * @param delta Trust region radius, note that this applies after scaling
 * transformation
 * @param theta parameter regulating stepback
 * @param lb lower optimization variable boundaries
 * @param ub upper optimization variable boundaries
 * @return New proposal steps
 */
std::vector<std::unique_ptr<Step>>
stepback_reflect(Step const &tr_step, DynamicVector<double> const &x,
                 DynamicVector<double> const &sg,
                 DynamicMatrix<double> const &hess,
                 CompressedMatrix<double> const &scaling,
                 CompressedMatrix<double> const &g_dscaling, const double delta,
                 const double theta, DynamicVector<double> const &lb,
                 DynamicVector<double> const &ub);

/**
 * @brief Compute new proposal steps according to a truncation strategy.
 * @param tr_step Reference trust region step that will be truncated
 * @param x Current values of the optimization variables
 * @param sg Rescaled objective function gradient at x
 * @param hess (Approximate) objective function Hessian at x
 * @param scaling Scaling transformation according to distance to boundary
 * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
 * transformation
 * @param delta Trust region radius, note that this applies after scaling
 * transformation
 * @param theta parameter regulating stepback
 * @param lb lower optimization variable boundaries
 * @param ub upper optimization variable boundaries
 * @return New proposal steps
 */
std::vector<std::unique_ptr<Step>> stepback_truncate(
    Step const &tr_step, DynamicVector<double> const &x,
    DynamicVector<double> const &sg, DynamicMatrix<double> const &hess,
    CompressedMatrix<double> const &scaling,
    CompressedMatrix<double> const &g_dscaling, const double delta,
    const double theta, DynamicVector<double> const &lb,
    DynamicVector<double> const &ub);

/**
 * @brief Refine a promising subset of the provided steps based on trust-constr
 * optimization
 * @param steps Reference trust region step that will be reflected
 * @param x Current values of the optimization variables
 * @param sg Rescaled objective function gradient at x
 * @param hess (Approximate) objective function Hessian at x
 * @param scaling Scaling transformation according to distance to boundary
 * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
 * transformation
 * @param delta Trust region radius, note that this applies after scaling
 * transformation
 * @param theta parameter regulating stepback
 * @param lb lower optimization variable boundaries
 * @param ub upper optimization variable boundaries
 * @return New proposal steps
 */
std::vector<std::unique_ptr<Step>>
stepback_refine(std::vector<std::unique_ptr<Step>> const &steps,
                DynamicVector<double> const &x, DynamicVector<double> const &sg,
                DynamicMatrix<double> const &hess,
                CompressedMatrix<double> const &scaling,
                CompressedMatrix<double> const &g_dscaling, const double delta,
                const double theta, DynamicVector<double> const &lb,
                DynamicVector<double> const &ub);

} // namespace fides

#endif // FIDES_STEPBACK_HPP
