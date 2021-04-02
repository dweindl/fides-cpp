#ifndef FIDES_CONSTANTS_HPP
#define FIDES_CONSTANTS_HPP

#include <chrono>
#include <cmath>
#include <map>
#include <string>

namespace fides {

/**
 * @brief The possible choices of subspace dimension in which the subproblem
 * will be solved.
 */
enum class SubSpaceDim {
    /** Two dimensional Newton/Gradient subspace */
    two,
    /** Full \f[ \mathbb{R}^n \f] */
    full
};

const std::map<SubSpaceDim, std::string> subspace_dim_to_str{
    {SubSpaceDim::two, "two"},
    {SubSpaceDim::full, "full"},
};

/**
 * @brief Possible choices of search refinement if proposed step
 * reaches optimization boundary
 */
enum class StepBackStrategy {
    /** single reflection at boundary */
    reflect_single,
    /** recursive reflections at boundary */
    reflect,
    /** truncate step at boundary and resolve subproblem */
    truncate,
    /** mix reflections and truncations */
    mixed,
};

const std::map<StepBackStrategy, std::string> step_back_strategy_str{
    {StepBackStrategy::reflect_single, "reflect_single"},
    {StepBackStrategy::reflect, "reflect"},
    {StepBackStrategy::truncate, "truncate"},
    {StepBackStrategy::mixed, "mixed"},
};

/**
 * @brief Options for the fides optimizers.
 */
struct Options {
    /** Maximum number of iterations allowed */
    int maxiter{1000};

    /** maximum amount of wall time in seconds */
    // TODO how long is max?
    std::chrono::seconds maxtime{std::chrono::seconds::max()};

    /** absolute tolerance for convergence based on objective value */
    double fatol{1e-8};

    /** relative tolerance for convergence based on objective value */
    double frtol{1e-8};

    /** tolerance for convergence based on x */
    double xtol{0};

    /** absolute tolerance for convergence based on grad */
    double gatol{1e-6};

    /** relative tolerance for convergence based on grad */
    double grtol{0};

    /** trust region subproblem subspace */
    SubSpaceDim subspace_solver = SubSpaceDim::full;

    /** method to use for stepback */
    StepBackStrategy stepback_strategy{StepBackStrategy::reflect};

    /** maximal fraction of step that would hit bounds */
    double theta_max{0.95};

    /** initial trust region radius */
    double delta_init{1.0};

    /** acceptance threshold for trust region ratio */
    double mu{0.25};

    /** trust region increase threshold for trust region ratio */
    double eta{0.75};

    /** factor by which trust region radius will be decreased */
    double gamma1{1.0 / 4.0};

    /** factor by which trust region radius will be increased */
    double gamma2{2.0};

    /** refine stepback */
    bool refine_stepback{false};
};

/**
 * @brief Possible statuses for the optimizer to indicate why
 * optimization exited.
 *
 * Negative value indicate errors while positive values indicate convergence.
 */
enum class ExitStatus {
    /** Optimizer did not run */
    did_not_run = 0,
    /** Reached maximum number of allowed iterations */
    max_iter = -1,
    /** Expected to reach maximum allowed time in next iteration */
    max_time = -2,
    /** Encountered non-finite fval/grad/hess */
    not_finite = -3,
    /** Exceeded specified boundaries */
    exceeded_boundary = -4,
    /** Trust region radius too small to proceed */
    delta_too_small = -5,
    /** Converged according to fval difference */
    ftol = 1,
    /** Converged according to x difference */
    xtol = 2,
    /** Converged according to gradient norm */
    gtol = 3
};

const std::map<ExitStatus, std::string> exit_status_to_str{
    {ExitStatus::did_not_run, "did_not_run"},
    {ExitStatus::max_iter, "max_iter"},
    {ExitStatus::max_time, "max_time"},
    {ExitStatus::not_finite, "not_finite"},
    {ExitStatus::exceeded_boundary, "exceeded_boundary"},
    {ExitStatus::delta_too_small, "delta_too_small"},
    {ExitStatus::ftol, "ftol"},
    {ExitStatus::xtol, "xtol"},
    {ExitStatus::gtol, "gtol"},
};

/** Minimum of two numbers, where NAN is considered the largest value. */
template <class T> const T &nanmin(const T &a, const T &b) {
    if (std::isnan(a))
        return b;
    if (std::isnan(b))
        return a;
    return (b < a) ? b : a;
}

} // namespace fides

#endif // FIDES_CONSTANTS_HPP
