#ifndef FIDES_MINIMIZE_HPP
#define FIDES_MINIMIZE_HPP

#include <fides/constants.hpp>
#include <fides/hessian_approximation.hpp>
#include <fides/steps.hpp>

#include <gsl/gsl-lite.hpp>

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

namespace fides {

using ::blaze::DynamicMatrix;
using ::blaze::DynamicVector;

/**
 * @brief Return type for objective functions
 *
 * The first value is the objective function value, the second is its gradient,
 * the third is its Hessian.
 */
using cost_fun_ret_t =
  std::tuple<double, DynamicVector<double>, DynamicMatrix<double>>;

/** Objective function type */
using cost_function_t = std::function<cost_fun_ret_t(DynamicVector<double>)>;

/**
 * @brief The fides optimizers.
 */
class Optimizer
{
  public:
    /**
     * @brief Create a new optimizer
     * @param cost_function The objective function. Expected to return a
     * tuple (fval, gradient, Hessian). If a hessian_approximation
     * is provided, the Hessian entry may be an empty Matrix.
     * @param lb Lower optimization boundaries. May contain -infinity.
     * @param ub Upper optimization boundaries. May contain infinity.
     * @param options Optimizer options and convergence criteria
     * @param hessian_approximation Class that performs the Hessian updates in
     * every iteration
     */
    Optimizer(cost_function_t const& cost_function,
              DynamicVector<double> const& lb,
              DynamicVector<double> const& ub,
              Options const& options,
              HessianApproximation* hessian_approximation);

    /**
     * @brief Minimize the objective function.
     *
     * Minimize the objective function the interior trust-region reflective
     * algorithm described by [ColemanLi1994] and [ColemanLi1996]. Convergence
     * with respect to function value is achieved when
     * \f[ |f_{k+1} - f_k| \f] < options.fatol - \f[ f_k \f] options.frtol.
     * Similarly, convergence with respect to optimization variables is achieved
     * when \f[ ||x_{k+1} - x_k|| \f]
     * < options.xatol - \f[ x_k \f] options.xrtol. Convergence with respect to
     * the gradient is achieved when \f[ ||g_k|| \f] < options.gatol or
     * `||g_k||` < options.grtol * `f_k`.
     * Other than that, optimization can be terminated when iterations exceed
     * options.maxiter or the elapsed time is expected to exceed
     * options.maxtime on the next iteration.
     *
     * @param x0 Initial guess
     * @return Tuple of:
     * * fval: final function value,
     * * x: final optimization variable values,
     * * grad: final gradient,
     * * hess: final Hessian (approximation)
     */
    std::tuple<double,
               DynamicVector<double>,
               DynamicVector<double>,
               DynamicMatrix<double>>
    minimize(DynamicVector<double> const& x0);

  private:
    /**
     * @brief Tracks the optimization variables that have minimal function
     * value independent of whether the step is accepted or not.
     * @param x_new New optimization variable values
     * @param fval_new Objective function value at x_new
     * @param grad_new Objective function gradient at x_new
     */
    void track_minimum(DynamicVector<double> const& x_new,
                       const double fval_new,
                       DynamicVector<double> const& grad_new);

    /**
     * @brief Update according to employed step
     * @param step Employed step
     * @param x_new New optimization variable values
     * @param fval_new Objective function value at x_new
     * @param grad_new Objective function gradient at x_new
     * @param hess_new (Approximate) objective function Hessian at x_new
     */
    void update(Step const& step,
                DynamicVector<double> const& x_new,
                const double fval_new,
                DynamicVector<double> const& grad_new,
                DynamicMatrix<double> const& hess_new);

    /**
     * @brief Update the trust region radius
     * @param fval new function value if step defined by step_sx is taken
     * @param grad new gradient value if step defined by step_sx is taken
     * @param step step
     * @param dv derivative of scaling vector v wrt x
     * @return `true` if the proposed step should be accepted, `false` otherwise
     */
    bool update_tr_radius(double const fval,
                          DynamicVector<double> const& grad,
                          Step const& step,
                          DynamicVector<double> const& dv);

    /**
     * @brief Check whether optimization has converged.
     * @param step update to optimization variables
     * @param fval updated objective function value
     * @param grad updated objective function gradient
     */
    void check_convergence(Step const& step,
                           double const fval,
                           DynamicVector<double> const& grad);

    /**
     * @brief Checks whether minimization should continue based on convergence,
     * iteration count and remaining computational budget.
     * @return `true` if minimization should continue, `false` otherwise.
     */
    bool check_continue();

    /**
     * @brief Ensures that x is non-degenerate.
     *
     * This should only be necessary for initial points.
     */
    void make_non_degenerate()
    {
        double eps = 1e2 * std::numeric_limits<double>::epsilon();
        make_non_degenerate(eps);
    }

    void make_non_degenerate(double eps);

    /**
     * @brief Computes the vector v and dv, the diagonal of its Jacobian.
     *
     * For the definition of v, see Definition 2 in [Coleman-Li1994].
     *
     * @return (v, dv): v scaling vector, dv diagonal of the Jacobian of v wrt x
     */
    std::tuple<DynamicVector<double>, DynamicVector<double>>
    get_affine_scaling();

    /**
     * @brief Prints diagnostic information about the current step to the log
     * @param accepted flag indicating whether the current step was accepted
     * @param step proposal step
     * @param fval new fval if step is accepted
     */
    void log_step(bool accepted, Step const& step, double fval) const;

    /**
     * @brief Prints diagnostic information about the initial step to the log
     */
    void log_step_initial() const;

    /**
     * @brief Prints the header for diagnostic information, should complement
     * `log_step`.
     */
    void log_header() const;

    /**
     * @brief Checks whether current objective function value, gradient and
     * Hessian (approximation) have finite values and optimization can continue.
     *
     * Throws std::runtime_error if any of the variables have non-finite
     * entries.
     */
    void check_finite() { check_finite(grad_, hess_); }

    /**
     * @brief Checks whether objective function value, gradient and Hessian
     * (approximation) have finite values and optimization can continue.
     *
     * \exception std::runtime_error
     * if any of the variables have non-finite entries.
     *
     * @param grad Gradient to be checked for finiteness.
     * @param hess Hessian (approximation) to be checked for finiteness.
     */
    void check_finite(DynamicVector<double> grad, DynamicMatrix<double> hess);

    void check_in_bounds() { check_in_bounds(x_); }

    /**
     * @brief Checks whether the current optimization variables are all within
     * the specified boundaries
     * \exception std::runtime_error if any of the variables are not within
     *  boundaries
     * @param x optimization variables
     */
    void check_in_bounds(DynamicVector<double> x);

    void reset();

    /** Objective function */
    cost_function_t cost_fun_;

  public:
    /** Optimizer options and convergence criteria */
    Options options_;

    /** Lower optimization boundaries */
    DynamicVector<double> lb_;

    /** Upper optimization boundaries */
    DynamicVector<double> ub_;

    /** Current optimization variables */
    DynamicVector<double> x_;

    /** Objective function gradient at x */
    DynamicVector<double> grad_;

    /** Objective function Hessian (approximation) at x */
    DynamicMatrix<double> hess_;

    /** Optimal optimization variables */
    DynamicVector<double> x_min_;

    /** Objective function gradient at x_min */
    DynamicVector<double> grad_min_;

    /** Hessian approximation */
    HessianApproximation* hessian_update_ = nullptr;

    /** Objective function value at x */
    double fval_{ std::numeric_limits<double>::infinity() };

    /** Objective function value at x_min */
    double fval_min_{ std::numeric_limits<double>::infinity() };

    /** Updated trust region radius */
    double delta_;

    /** Trust region radius that was used for the current step */
    double delta_iter_;

    /** Ratio of expected and actual improvement */
    double tr_ratio_{ 1.0 };

    /** Current iteration */
    int iteration_{ 0 };

    /** Flag indicating whether optimization has converged */
    bool converged_{ false };

    /** Exit status */
    ExitStatus exit_flag_{ ExitStatus::did_not_run };

    /** Time at which optimization was started */
    std::chrono::time_point<std::chrono::system_clock> starttime_;
};

} // namespace fides

#endif // FIDES_MINIMIZE_HPP
