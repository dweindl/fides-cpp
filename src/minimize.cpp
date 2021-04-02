#include "fides/minimize.hpp"

#include <fides/trust_region.hpp>

#include "spdlog/spdlog.h"

#include <exception>
#include <iostream>
#include <stdexcept>

using ::blaze::abs;
using ::blaze::min;
using ::blaze::outer;
using ::blaze::trans;

namespace fides {

Optimizer::Optimizer(const cost_function_t &cost_function,
                     const DynamicVector<double> &lb,
                     const DynamicVector<double> &ub,
                     const Options &options,
                     fides::HessianApproximation* hessian_approximation)
  : cost_fun_(cost_function)
  , options_(options)
  , lb_(lb)
  , ub_(ub)
  , hessian_update_(hessian_approximation)
  , starttime_(std::chrono::milliseconds::zero())

{
    if (lb_.size() != ub_.size()) {
        throw std::runtime_error(
          "Length of provided lower and upper bounds do not match (" +
          std::to_string(lb_.size()) + " vs. " + std::to_string(ub_.size()) +
          ")!");
    }

    delta_ = options_.delta_init;
    delta_iter_ = delta_;
    x_ = DynamicVector<double>(ub.size());
    grad_ = DynamicVector<double>(ub.size());
    hess_ = DynamicMatrix<double>(ub.size(), ub.size());
    x_min_ = x_;
    grad_min_ = grad_;

    spdlog::set_level(spdlog::level::debug);
}

std::tuple<double,
           DynamicVector<double>,
           DynamicVector<double>,
           DynamicMatrix<double>>
Optimizer::minimize(const DynamicVector<double>& x0)
{
    if (x0.size() != lb_.size()) {
        throw std::runtime_error("Provided x0 has different length (" +
                                 std::to_string(x0.size()) +
                                 " than the provided parameter bounds (" +
                                 std::to_string(lb_.size()) + ")!");
    }

    reset();
    x_ = x0;
    make_non_degenerate();
    check_in_bounds();

    std::tie(fval_, grad_, hess_) = cost_fun_(x_);

    if (hessian_update_) {
        auto hybrid = dynamic_cast<HybridUpdate*>(hessian_update_);
        if(hybrid && hybrid->init_with_hess_)
            hybrid->set_init(hess_);
        hessian_update_->init_mat(x_.size());
        hess_ = hessian_update_->get_mat();
    }

    if (grad_.size() != x_.size()) {
        throw std::runtime_error("Provided objective function must return a "
                                 "gradient vector of the same shape as x, "
                                 "x has " +
                                 std::to_string(x_.size()) +
                                 " entries but gradient has " +
                                 std::to_string(grad_.size()) + "!");
    }

    // hessian approximation would error on these earlier
    if (hess_.rows() != hess_.columns()) {
        throw std::runtime_error("Provided objective function must return a "
                                 "square Hessian matrix!");
    }

    if (hess_.rows() != x_.size()) {
        throw std::runtime_error(
          "Provided objective function must return a "
          "square Hessian matrix with same dimension as x. "
          "x has " +
          std::to_string(x_.size()) + " entries but Hessian has " +
          std::to_string(hess_.rows()) + "!");
    }

    track_minimum(x_, fval_, grad_);
    log_header();
    log_step_initial();

    check_finite();
    converged_ = false;

    double fval_new;
    DynamicVector<double> grad_new;
    DynamicMatrix<double> hess_new;

    while (check_continue()) {
        iteration_ += 1;
        delta_iter_ = delta_;

        auto [v, dv] = get_affine_scaling();

        auto scaling = blaze::CompressedMatrix<double>(v.size(), v.size());
        auto diag(blaze::band(scaling, 0L));
        diag = blaze::sqrt(abs(v));
        auto theta =
          std::max(options_.theta_max, 1 - blaze::maxNorm(v * grad_));
        auto step = trust_region(x_,
                                 grad_,
                                 hess_,
                                 scaling,
                                 delta_iter_,
                                 dv,
                                 theta,
                                 lb_,
                                 ub_,
                                 options_.subspace_solver,
                                 options_.stepback_strategy,
                                 options_.refine_stepback);

        auto x_new = x_ + step->s_ + step->s0_;

        std::tie(fval_new, grad_new, hess_new) = cost_fun_(x_new);

        if (std::isfinite(fval_new)) {
            check_finite(grad_new, hess_new);
        }
        auto accepted = update_tr_radius(fval_new, grad_new, *step, dv);

        if (iteration_ % 10 == 0) {
            log_header();
        }
        log_step(accepted, *step, fval_new);
        check_convergence(*step, fval_new, grad_new);
        track_minimum(x_new, fval_new, grad_new);

        if (accepted) {
            update(*step, x_new, fval_new, grad_new, hess_new);
        }
    }
    spdlog::info("Finished: fval: {:.3E} ||g||: {:.3E} exit: {}",
                 fval_,
                 norm(grad_),
                 exit_status_to_str.at(exit_flag_));
    return std::make_tuple(fval_, x_, grad_, hess_);
}

void
Optimizer::track_minimum(const DynamicVector<double>& x_new,
                         const double fval_new,
                         const DynamicVector<double>& grad_new)
{
    if (std::isfinite(fval_new) && fval_new < fval_min_) {
        x_min_ = x_new;
        fval_min_ = fval_new;
        grad_min_ = grad_new;
    }
}

void
Optimizer::update(const Step& step,
                  const DynamicVector<double>& x_new,
                  const double fval_new,
                  const DynamicVector<double>& grad_new,
                  const DynamicMatrix<double>& hess_new)
{
    if (hessian_update_) {
        hessian_update_->update(step.s_ + step.s0_, grad_new - grad_);
    }

    if (!hessian_update_ ||
        (dynamic_cast<HybridUpdate*>(hessian_update_) &&
         iteration_ <
           dynamic_cast<HybridUpdate*>(hessian_update_)->switch_iteration_)) {
        hess_ = hess_new;
    } else {
        hess_ = hessian_update_->get_mat();
    }
    check_in_bounds(x_new);
    fval_ = fval_new;
    x_ = x_new;
    grad_ = grad_new;
    check_finite();
    make_non_degenerate();
}

bool
Optimizer::update_tr_radius(const double fval,
                            const DynamicVector<double>& grad,
                            const Step& step,
                            const DynamicVector<double>& dv)
{
    auto stepsx = step.ss_ + step.ss0_;
    auto nsx = norm(stepsx);

    if (not std::isfinite(fval)) {
        tr_ratio_ = 0.0;

        delta_ = nanmin(delta_ * options_.gamma1, nsx / 4.0);
        return false;
    }
    double qpval = 0.5 * dot(stepsx, (dv * abs(grad) * stepsx));
    tr_ratio_ = (fval + qpval - fval_) / step.qpval_;
    auto interior_solution = nsx < delta_iter_ * 0.9;

    // values as proposed in algorithm 4.1 in Nocedal & Wright
    if (tr_ratio_ >= options_.eta && !interior_solution) {
        // increase radius
        delta_ = options_.gamma2 * delta_;
    } else if (tr_ratio_ <= options_.mu) {
        // decrease radius
        delta_ = nanmin(delta_ * options_.gamma1, nsx / 4.0);
    }
    return tr_ratio_ > 0.0;
}

void
Optimizer::check_convergence(const Step& step,
                             const double fval,
                             const DynamicVector<double>& grad)
{
    bool converged = false;

    auto gnorm = norm(grad);
    auto stepsx = step.ss_ + step.ss0_;
    auto nsx = norm(stepsx);

    if (delta_ <= delta_iter_ &&
        abs(fval - fval_) < options_.fatol + options_.frtol * abs(fval_)) {
        exit_flag_ = ExitStatus::ftol;
        spdlog::warn("Stopping as function difference "
                     "{:.2E} was smaller than specified "
                     "tolerances (atol={:.2E}, rtol={:.2E})",
                     std::abs(fval_ - fval),
                     options_.fatol,
                     options_.frtol);

        converged = true;
    } else if (iteration_ > 1 && nsx < options_.xtol) {
        exit_flag_ = ExitStatus::xtol;
        spdlog::warn("Stopping as norm of step "
                     "{} was smaller than specified "
                     "tolerance (tol={:.2E})",
                     nsx,
                     options_.xtol);
        converged = true;
    } else if (gnorm <= options_.gatol) {
        exit_flag_ = ExitStatus::gtol;
        spdlog::warn("Stopping as gradient norm satisfies absolute convergence "
                     "criteria: {:.2E} < {:.2E}",
                     gnorm,
                     options_.gatol);
        converged = true;
    } else if (gnorm <= options_.grtol * fval_) {
        exit_flag_ = ExitStatus::gtol;
        spdlog::warn("Stopping as gradient norm satisfies relative convergence "
                     "criteria: {:.2E} < {:.2E} * {:.2E}",
                     gnorm,
                     options_.grtol,
                     fval_);
        converged = true;
    }
    converged_ = converged;
}

bool
Optimizer::check_continue()
{
    if (converged_)
        return false;

    if (iteration_ >= options_.maxiter) {
        exit_flag_ = ExitStatus::max_iter;
        spdlog::warn("Stopping as maximum number of iterations {} "
                     "was exceeded.",
                     options_.maxiter);
        return false;
    }

    std::chrono::duration<double> time_elapsed =
      (std::chrono::system_clock::now() - starttime_);
    auto time_remaining = options_.maxtime - time_elapsed;
    auto avg_iter_time = time_elapsed / (iteration_ + (iteration_ == 0));
    if (time_remaining < avg_iter_time) {
        exit_flag_ = ExitStatus::max_time;
        spdlog::warn("Stopping as maximum runtime {}s is expected to be "
                     "exceeded in the next iteration.",
                     options_.maxtime.count());
        return false;
    }

    if (delta_ < std::numeric_limits<double>::epsilon()) {
        exit_flag_ = ExitStatus::delta_too_small;
        spdlog::warn("Stopping as trust region radius {:.2E} is "
                     "smaller than machine precision.",
                     delta_);
        return false;
    }
    return true;
}

void
Optimizer::make_non_degenerate(double eps)
{
    if (min(abs(ub_ - x_)) < eps || min(abs(x_ - lb_)) < eps) {
        x_ = blaze::map(x_, ub_, lb_, [eps](double x, double ub, double lb) {
            if (ub - x < eps)
                return x - eps;
            if (x - lb < eps)
                return x + eps;
            return x;
        });
    }
}

std::tuple<DynamicVector<double>, DynamicVector<double>>
Optimizer::get_affine_scaling()
{
    // this implements no scaling for variables that are not constrained by
    // bounds ((iii) and (iv) in Definition 2)

    DynamicVector<double> v = blaze::forEach(
      grad_, [](auto grad) { return blaze::sign(grad) + (grad == 0); });
    DynamicVector<double> dv = blaze::zero<double>(x_.size());

    // this implements scaling for variables that are constrained by
    // bounds ( i and ii in Definition 2) bounds is equal to lb if grad <
    // 0 ub if grad >= 0
    auto bounds = ((1 + v) * lb_ + (1 - v) * ub_) / 2;
    DynamicVector<bool> bounded =
      blaze::forEach(bounds, [](auto bound) { return std::isfinite(bound); });
    v = blaze::map(
      v, x_, bounds, bounded, [](auto v, auto x, auto bound, auto bounded) {
          if (bounded)
              return x - bound;
          return v;
      });
    dv = blaze::map(dv, bounded, [](auto dv, auto bounded) {
        if (bounded)
            return 1.0;
        return dv;
    });

    return std::make_tuple(v, dv);
}

void
Optimizer::log_step(bool accepted, const Step& step, double fval) const
{
    auto normdx = norm(step.s_ + step.s0_);
    auto iter_width =
      std::max<long>(std::to_string(options_.maxiter).size(), 5);
    if (!std::isfinite(fval)) {
        fval = fval_;
    }

    spdlog::info("{0: >{1}} | {2:+.3E} | "
                 "{3:+.2E} | {4:+.2E} | {5:+.2E} | {6:+.2E} | {7:+.2E} | "
                 "{8:.2E} | {9:.2E} | {10:.2E} | {11: >4} | {12: >4} | "
                 "{13: >4} | {14}",
                 iteration_,
                 iter_width,
                 accepted ? fval : fval_,
                 fval - fval_,
                 step.qpval_,
                 tr_ratio_,
                 delta_iter_,
                 norm(grad_),
                 normdx,
                 step.theta_,
                 step.alpha_,
                 step.type,
                 step.reflection_count_,
                 step.truncation_count_,
                 int(accepted));
}

void
Optimizer::log_step_initial() const
{
    auto iter_width =
      std::max<long>(std::to_string(options_.maxiter).size(), 5);

    spdlog::info("{0: >{1}} | {2:+.3E} |    NaN    |    NaN    |    NaN   "
                 " | {3:+.2E} | {4:+.2E} |   NaN    |   NaN    |   NaN   "
                 " |  NaN |  NaN |  NaN | {5}",
                 iteration_,
                 iter_width,
                 fval_,
                 delta_,
                 norm(grad_),
                 int(std::isfinite(fval_)));
}

void
Optimizer::log_header() const
{
    auto iter_width =
      std::max<long>(std::to_string(options_.maxiter).size(), 5);

    spdlog::info("{: <{}} |    fval    | fval diff | pred diff | tr ratio  "
                 "|  delta    |  ||g||   | ||step|| |  theta   |  alpha   "
                 "| step | refl | trun | accept",
                 "",
                 iter_width);
}

void
Optimizer::check_finite(DynamicVector<double> grad, DynamicMatrix<double> hess)
{
    std::string pointstr;
    if (iteration_ == 0) {
        pointstr = "at initial point.";
    } else {
        pointstr =
          std::string("at iteration ") + std::to_string(iteration_) + ".";
    }

    if (!std::isfinite(fval_)) {
        exit_flag_ = ExitStatus::not_finite;
        throw std::runtime_error("Encountered non-finite function value " +
                                 std::to_string(fval_) + " " + pointstr);
    }

    if (!blaze::isfinite(grad)) {
        exit_flag_ = ExitStatus::not_finite;

        std::stringstream ss;
        for (std::size_t i = 0; i < grad.size(); ++i)
            if (!std::isfinite(grad[i]))
                ss << (i ? ", " : "") << i;

        throw std::runtime_error(
          "Encountered non-finite gradient entries for indices " + ss.str() +
          " " + pointstr);
    }
    if (!blaze::isfinite(hess)) {
        exit_flag_ = ExitStatus::not_finite;

        std::stringstream ss;
        for (std::size_t i = 0; i < hess.rows(); ++i)
            for (std::size_t j = 0; j < hess.columns(); ++j)
                if (!std::isfinite(hess(i, j)))
                    ss << (i ? ", " : "") << "(" << i << "," << j << ")";

        throw std::runtime_error(
          "Encountered non-finite gradient hessian for indices " + ss.str() +
          " " + pointstr);
    }
}

void
Optimizer::check_in_bounds(DynamicVector<double> x)
{
    std::string pointstr;
    if (iteration_ == 0) {
        pointstr = " at initial point.";
    } else {
        pointstr =
          std::string(" at iteration ") + std::to_string(iteration_) + ".";
    }

    auto fun = [&x, this, &pointstr](DynamicVector<double> ref,
                                     double sign,
                                     std::string const& name) {
        auto diff = sign * (ref - x);
        DynamicVector<bool> exceeded_bounds = blaze::forEach(diff, [](double cur_diff) {
            // out of bounds?
            return cur_diff > 0.0;
        });

        if (blaze::sum(exceeded_bounds) > 0) {
            std::stringstream ss_indices;
            std::stringstream ss_diff;
            for (std::size_t i = 0; i < exceeded_bounds.size(); ++i) {
                if (exceeded_bounds[i]) {
                    ss_indices << (i ? ", " : "") << i;
                    ss_diff << (i ? ", " : "") << diff[i];
                }
            }
            exit_flag_ = ExitStatus::exceeded_boundary;
            throw std::runtime_error("Exceeded " + name + " for indices " +
                                     ss_indices.str() + " by " + ss_diff.str()
                                     + " " + pointstr);
        }
    };
    fun(ub_, -1.0, "upper bounds");
    fun(lb_, 1.0, "lower bounds");
}

void
Optimizer::reset()
{
    starttime_ = std::chrono::system_clock::now();
    delta_ = options_.delta_init;
    delta_iter_ = delta_;
    iteration_ = 0;
    converged_ = false;
    fval_min_ = std::numeric_limits<double>::infinity();
}

} // namespace fides
