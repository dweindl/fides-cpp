#include "fides/trust_region.hpp"

#include <exception>
#include <iomanip>
#include <iostream>

#include "fides/stepback.hpp"
#include "fides/steps.hpp"

#include "spdlog/spdlog.h"

namespace fides {

std::unique_ptr<Step>
trust_region(const DynamicVector<double>& x,
             const DynamicVector<double>& g,
             const DynamicMatrix<double>& hess,
             const CompressedMatrix<double>& scaling,
             const double delta,
             const DynamicVector<double>& dv,
             const double theta,
             const DynamicVector<double>& lb,
             const DynamicVector<double>& ub,
             const SubSpaceDim subspace_dim,
             const StepBackStrategy stepback_strategy,
             const bool refine_stepback)
{
    auto sg = scaling * g;
    auto g_dscaling = CompressedMatrix<double>(g.size(), g.size());
    blaze::band(g_dscaling, 0L) = abs(g) * dv;

    std::vector<std::unique_ptr<Step>> steps;

    switch (subspace_dim) {
        case SubSpaceDim::two:
            steps.emplace_back(std::make_unique<TRStep2D>(
              x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub));
            break;
        case SubSpaceDim::full:
            steps.emplace_back(std::make_unique<TRStepFull>(
              x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub));
            break;
        default:
            throw std::runtime_error("Invalid choice of subspace dimension.");
    }

    auto& tr_step = *steps.front();

    tr_step.calculate();

    /*
     * In case of truncation, we hit the boundary and we check both the
     * gradient and the reflected step, either of which could be better than
     * the TR step
     */
    if (tr_step.alpha_ < 1.0 && g.size() > 1) {
        // save before clearing, tr_step will stay valid
        auto tr_step_tmp = std::move(steps.front());
        steps.clear();
        steps.emplace_back(std::make_unique<GradientStep>(
          x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub));
        steps.back()->calculate();

        if (stepback_strategy == StepBackStrategy::reflect_single) {
            steps.emplace_back(std::make_unique<TRStepReflected>(
              x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub, tr_step));
            steps.back()->calculate();
        }

        if (stepback_strategy == StepBackStrategy::reflect ||
            stepback_strategy == StepBackStrategy::mixed) {
            auto tmp = stepback_reflect(
              tr_step, x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub);
            std::move(
              std::begin(tmp), std::end(tmp), std::back_inserter(steps));
        }

        if (stepback_strategy == StepBackStrategy::truncate ||
            stepback_strategy == StepBackStrategy::mixed) {
            auto tmp = stepback_truncate(
              tr_step, x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub);
            std::move(
              std::begin(tmp), std::end(tmp), std::back_inserter(steps));
        }
        if (refine_stepback) {
            auto tmp = stepback_refine(
              steps, x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub);
            std::move(
              std::begin(tmp), std::end(tmp), std::back_inserter(steps));
        }
    }
    if (steps.size() > 1) {
        std::stringstream ss;
        for (auto const& step : steps) {
            if (step != *steps.begin())
                ss << " | ";

            ss << step->type
               << (step->reflection_count_ > 0
                     ? std::to_string(step->reflection_count_)
                     : "")
               << std::scientific << std::setprecision(2)
               << ": [qp: " << step->qpval_ << ", a: " << step->alpha_ << "]";
        }
        spdlog::debug(ss.str());
    }

    auto min_qpval_step = std::min_element(
      steps.begin(), steps.end(), [](auto const& a, auto const& b) {
          return a->qpval_ < b->qpval_;
      });

    return std::move(*min_qpval_step);
}

} // namespace fides
