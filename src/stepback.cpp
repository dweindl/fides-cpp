#include "fides/stepback.hpp"

#include "fides/minimize.hpp" // nanmin

#include <blaze/Math.h>

#include <exception>
#include <iostream>

using blaze::outer;
using blaze::trans;

namespace fides {

std::vector<std::unique_ptr<Step>> stepback_truncate(
    Step const &tr_step, DynamicVector<double> const &x,
    DynamicVector<double> const &sg, DynamicMatrix<double> const &hess,
    CompressedMatrix<double> const &scaling,
    CompressedMatrix<double> const &g_dscaling, const double delta,
    const double theta, DynamicVector<double> const &lb,
    DynamicVector<double> const &ub) {
    std::vector<std::unique_ptr<Step>> steps;
    steps.emplace_back(std::make_unique<TRStepTruncated>(
        x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub, tr_step));
    auto rtt_step = steps.back().get();
    rtt_step->calculate();

    while (rtt_step->subspace_.columns() > 0) {
        if (rtt_step->alpha_ == 1.0)
            break;

        steps.emplace_back(std::make_unique<TRStepTruncated>(
            x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub, *rtt_step));
        rtt_step = steps.back().get();
        rtt_step->calculate();
    }

    return steps;
}

std::vector<std::unique_ptr<Step>>
stepback_reflect(Step const &tr_step, DynamicVector<double> const &x,
                 DynamicVector<double> const &sg,
                 DynamicMatrix<double> const &hess,
                 CompressedMatrix<double> const &scaling,
                 CompressedMatrix<double> const &g_dscaling, const double delta,
                 const double theta, DynamicVector<double> const &lb,
                 DynamicVector<double> const &ub) {
    std::vector<std::unique_ptr<Step>> steps;
    steps.emplace_back(std::make_unique<TRStepReflected>(
        x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub, tr_step));
    auto rtr_step = steps.back().get();
    rtr_step->calculate();
    for (size_t ireflection = 0; ireflection < x.size() - 1; ++ireflection) {
        if (rtr_step->alpha_ == 1.0)
            break;

        // recursively add more reflections
        steps.emplace_back(std::make_unique<TRStepReflected>(
            x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub, *rtr_step));
        rtr_step = steps.back().get();
        rtr_step->calculate();
    }
    return steps;
}

std::vector<std::unique_ptr<Step>>
stepback_refine(std::vector<std::unique_ptr<Step>> const &steps,
                DynamicVector<double> const &x, DynamicVector<double> const &sg,
                DynamicMatrix<double> const &hess,
                CompressedMatrix<double> const &scaling,
                CompressedMatrix<double> const &g_dscaling, const double delta,
                const double theta, DynamicVector<double> const &lb,
                DynamicVector<double> const &ub) {
    Expects(!steps.empty());
    double min_qpval = steps.front()->qpval_;
    for (auto const &step : steps) {
        min_qpval = nanmin(min_qpval, step->qpval_);
    }

    // TODO update to 0.3.4
    std::vector<std::unique_ptr<Step>> ref_steps;
    for (auto const &step : steps) {
        if ((step->alpha_ == 1.0 &&
             (step->type != "trnd" && step->type != "tr2d" &&
              step->type != "grad")) ||
            (step->alpha_ < 1.0 && step->qpval_ < min_qpval / 2.0)) {
            ref_steps.emplace_back(std::make_unique<RefinedStep>(
                x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub, *step));
            ref_steps.back()->calculate();
        }
    }
    return ref_steps;
}

} // namespace fides
