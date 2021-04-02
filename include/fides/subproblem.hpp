/**
 * Subproblem Solvers
 * ------------------
 *
 * This module provides the machinery to solve 1- and N-dimensional
 * trust-region subproblems.
 */

#ifndef FIDES_SUBPROBLEM_HPP
#define FIDES_SUBPROBLEM_HPP

#include <blaze/math/Forward.h>
#include <functional>

using blaze::DynamicMatrix;
using blaze::DynamicVector;

namespace fides {
/**
 * @brief Solve the special case of a one-dimensional subproblem
 * @param B Hessian of the quadratic subproblem
 * @param g Gradient of the quadratic subproblem
 * @param s Vector defining the one-dimensional search direction
 * @param delta Norm boundary for the solution of the quadratic subproblem
 * @param s0 Reference point from where search is started, also counts towards
 * norm of step
 * @return Proposed step-length
 */
DynamicVector<double>
solve_1d_trust_region_subproblem(DynamicMatrix<double> const& B,
                                 DynamicVector<double> const& g,
                                 DynamicVector<double> const& s,
                                 const double delta,
                                 DynamicVector<double> const& s0);

/**
 * @brief Newton-Raphson method for finding roots in a single-variable real
 * valued function.
 * @param fun Function of which to find roots
 * @param funprime Derivative of `fun`
 * @param x0 Initial guess for root
 * @param atol Absolute tolerance
 * @param maxiter Maximum number of iterations
 * @return Approximation of the root of `fun`
 */
double
newton(std::function<double(double)> const& fun,
       std::function<double(double)> const& funprime,
       const double x0,
       const double atol,
       const int maxiter);

/**
 * @brief Brent's method for finding roots in a single-variable real
 * valued function on an interval [a, b].
 *
 * See https://en.wikipedia.org/wiki/Brent%27s_method
 *
 * @param fun Function of which to find roots
 * @param a lower end of the interval
 * @param b upper end of the interval
 * @param xtol Absolute tolerance on the root
 * @param maxiter Maximum number of iterations
 * @return Approximation of the root of `fun`
 */
double
brentq(std::function<double(double)> const& fun,
       const double a,
       const double b,
       const double xtol,
       const int maxiter);

/**
 * @brief Exactly solve the n-dimensional subproblem.
 *
 * \f[ argmin_s\{s^T B s + s^T g = 0: ||s|| <= \Delta, s \in \mathbb{R}^n\} \f]
 *
 * The solution is characterized by the equation \f[ -(B + \lambda I)s = g \f].
 *
 * If B is positive definite, the solution can be obtained by
 * \f[ \lambda = 0 \f] if \f[ Bs = -g \f] satisfies \f[ ||s|| <= \Delta \f].
 *
 * If B is indefinite or \f[ Bs = -g \f] satisfies \f[ ||s|| > \Delta \f] and
 * an appropriate \f[ \lambda \f] has to be identified via 1D root finding of
 * the secular equation
 *
 * \f[ \phi(\lambda) = \frac{1}{||s(\lambda)||} - \frac{1}{\Delta} = 0 \f]
 *
 * with \f[ s(\lambda) \f] computed according to an eigenvalue decomposition
 * of B. The eigenvalue decomposition, although being more expensive than a
 * Cholesky decomposition, has the advantage that eigenvectors are invariant to
 * changes in \f[ \lambda \f] and eigenvalues are linear in \f[ \lambda \f],
 * so factorization only has to be performed once. We perform the linesearch
 * via Newton's algorithm and Brent-Q as fallback.
 *
 * The hard case is treated separately and serves as general fallback.
 *
 * @param B Hessian of the quadratic subproblem
 * @param g Gradient of the quadratic subproblem
 * @param delta Norm boundary for the solution of the quadratic subproblem
 * @return (s, step_type):
 * s: Selected step,
 * step_type: Type of solution that was obtained
 */
std::tuple<DynamicVector<double>, std::string>
solve_nd_trust_region_subproblem(DynamicMatrix<double> const& B,
                                 DynamicVector<double> const& g,
                                 double delta);

/**
 * @brief Computes the solution \f[ s(\lambda) \f] as subproblem solution
 * according to \f[ -(B + \lambda I)s = g \f].
 * @param lam \f[ \lambda \f]
 * @param w precomputed eigenvector coefficients for -g
 * @param eigvals precomputed eigenvalues of B
 * @param eigvecs precomputed eigenvectors of B
 * @return \f[ s(\lambda) \f]
 */
DynamicVector<double>
slam(const double lam,
     const DynamicVector<double>& w,
     const DynamicVector<double>& eigvals,
     const DynamicMatrix<double>& eigvecs);

/**
 * @brief Computes the derivative of the solution \f[ s(\lambda) \f] with
 * respect to lambda, where \f[ s \f] is the subproblem solution
 * according to \f[ -(B + \lambda I)s = g \f].
 *
 * @param lam \f[ \lambda \f]
 * @param w precomputed eigenvector coefficients for -g
 * @param eigvals precomputed eigenvalues of B
 * @param eigvecs precomputed eigenvectors of B
 * @return \f[ \frac{\partial s(\lambda)}{\partial \lambda} \f]
 */
DynamicVector<double>
dslam(const double lam,
      const DynamicVector<double>& w,
      const DynamicVector<double>& eigvals,
      const DynamicMatrix<double>& eigvecs);

/**
 * @brief Secular equation
 *
 * \f[ \phi(\lambda) = \frac{1}{||s||} - \frac{1}{\Delta} \f]
 *
 * Subproblem solutions are given by the roots of this equation.
 *
 * @param lam \f[ \lambda \f]
 * @param w precomputed eigenvector coefficients for -g
 * @param eigvals precomputed eigenvalues of B
 * @param eigvecs precomputed eigenvectors of B
 * @param delta trust region radius \f[ \Delta \f]
 * @return \f[ \phi(\lambda) \]
 */
double
secular(const double lam,
        const DynamicVector<double>& w,
        const DynamicVector<double>& eigvals,
        const DynamicMatrix<double>& eigvecs,
        const double delta);

/**
 * @brief Derivative of the secular equation
 *
 * \f[ \phi(\lambda) = \frac{1}{||s||} - \frac{1}{\Delta} \f]
 *
 * with respect to \f[ \lambda \f]
 *
 * @param lam \f[ \lambda \f]
 * @param w precomputed eigenvector coefficients for -g
 * @param eigvals precomputed eigenvalues of B
 * @param eigvecs precomputed eigenvectors of B
 * @return \f[ \frac{\partial \phi(\lambda)}{\partial \lambda} \]
 */
double
dsecular(double const lam,
         DynamicVector<double> const& w,
         DynamicVector<double> const& eigvals,
         DynamicMatrix<double> const& eigvecs);

} // namespace fides

#endif // FIDES_SUBPROBLEM_HPP
