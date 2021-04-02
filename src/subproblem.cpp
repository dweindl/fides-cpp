#include "fides/subproblem.hpp"

#include "spdlog/spdlog.h"

#include <blaze/Math.h>

#include <gsl/gsl-lite.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>

using blaze::norm;
using blaze::trans;

namespace fides {

DynamicVector<double>
solve_1d_trust_region_subproblem(DynamicMatrix<double> const& B,
                                 DynamicVector<double> const& g,
                                 DynamicVector<double> const& s,
                                 const double delta,
                                 DynamicVector<double> const& s0)
{
    if (delta == 0.0)
        return DynamicVector<double>(1U, 0.0);

    auto a = 0.5 * blaze::dot((B * s), s);
    auto b = trans(s) * g;

    auto minq = -b / (2 * a);
    double tau;
    if (a > 0.0 && blaze::eval(norm(minq * s + s0)) <= delta) {
        // interior solution
        tau = minq;
    } else {
        auto nrms0 = norm(s0);
        if (nrms0 == 0) {
            tau = -delta * blaze::sign(b);
        } else if (nrms0 >= delta) {
            tau = 0;
        } else {
            tau =
              brentq([s, s0, delta](
                       auto q) { return 1.0 / norm(q * s + s0) - 1.0 / delta; },
                     0.0,
                     2.0 * delta,
                     1e-12,
                     100);
        }
    }
    return DynamicVector<double>(1U, tau);
}

double
newton(std::function<double(double)> const& fun,
       std::function<double(double)> const& funprime,
       const double x0,
       const double atol,
       const int maxiter)
{
    Expects(atol >= 0.0);
    Expects(maxiter > 0);

    auto x_cur = x0;

    for (int iteration = 0; iteration < maxiter; ++iteration) {
        auto fval = fun(x_cur);
        if (fval == 0.0)
            // root found
            return x_cur;

        auto fder = funprime(x_cur);
        if (fder == 0.0) {
            throw std::runtime_error(
              "Derivative was zero at " + std::to_string(x_cur) +
              ". Failed to converge.");
        }

        auto newton_step = fval / fder;
        auto x_next = x_cur - newton_step;
        if (std::fabs(x_next - x_cur) < atol)
            return x_next;
        x_cur = x_next;
    }
    return x_cur;
}

double
brentq(const std::function<double(double)>& fun,
       const double a,
       const double b,
       const double xtol,
       const int maxiter)
{
    // Modified version of
    // https://github.com/scipy/scipy/blob/5f4c4d802e5a56708d86909af6e5685cd95e6e66/scipy/optimize/Zeros/brentq.c
    // which was released under the following terms:
    //
    // Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    // All rights reserved.
    //
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the following conditions
    // are met:
    //
    // 1. Redistributions of source code must retain the above copyright
    //    notice, this list of conditions and the following disclaimer.
    //
    // 2. Redistributions in binary form must reproduce the above
    //    copyright notice, this list of conditions and the following
    //    disclaimer in the documentation and/or other materials provided
    //    with the distribution.
    //
    // 3. Neither the name of the copyright holder nor the names of its
    //    contributors may be used to endorse or promote products derived
    //    from this software without specific prior written permission.
    //
    // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    // "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    // LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    // A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    // OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    // SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    // LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    // DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    // THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Expects(xtol >= 0.0);
    Expects(maxiter > 0);
    Expects(b > a);

    // previous estimate
    auto xpre = a;
    auto fpre = fun(xpre);
    // current estimate
    auto xcur = b;
    auto fcur = fun(xcur);

    if (fpre * fcur > 0.0) {
        std::runtime_error("Root is not bracketed.");
    }

    // starting at root?
    if (fpre == 0.0) {
        // converged
        return xpre;
    }
    if (fcur == 0.0) {
        // converged
        return xcur;
    }

    // tolerance is 2*delta
    const double delta = xtol / 2.0;

    auto xblk = 0.0;
    auto fblk = 0.0;
    auto spre = 0.0;
    auto scur = 0.0;

    for (int iteration = 0; iteration < maxiter; ++iteration) {
        if (fpre * fcur < 0) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        // bisection
        auto sbis = (xblk - xcur) / 2.0;

        if (fcur == 0.0 || fabs(sbis) < delta) {
            // converged
            return xcur;
        }

        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            double stry;

            if (xpre == xblk) {
                // secant method
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                // inverse quadratic interpolation
                auto dpre = (fpre - fcur) / (xpre - xcur);
                auto dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) /
                       (dblk * dpre * (fblk - fpre));
            }
            if (2 * fabs(stry) < std::min(fabs(spre), 3 * fabs(sbis) - delta)) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;

        if (fabs(scur) > delta) {
            xcur += scur;
        } else {
            xcur += (sbis > 0 ? delta : -delta);
        }

        fcur = fun(xcur);
    }
    // not converged
    return xcur;
}

std::tuple<DynamicVector<double>, std::string>
solve_nd_trust_region_subproblem(const DynamicMatrix<double>& B,
                                 const DynamicVector<double>& g,
                                 double delta)
{
    if (delta == 0) {
        return { blaze::zero<double>(g.size()), "zero" };
    }

    // See Nocedal & Wright 2006 for details

    // INITIALIZATION
    // instead of a cholesky factorization, we go with an eigenvalue
    // decomposition, which works pretty well for n=2
    DynamicVector<std::complex<double>> eigvals_c(B.rows());
    DynamicMatrix<std::complex<double>> eigvecs_c(B.rows(), B.rows());
    eigen(B, eigvals_c, eigvecs_c);
    auto eigvals = real(eigvals_c);
    // if B is row-major, eigenvectors are rows
    auto eigvecs = trans(real(eigvecs_c));
    DynamicVector<double> w = -trans(eigvecs) * g;
    auto jmin = argmin(eigvals);
    auto mineig = eigvals[jmin];

    // since B symmetric eigenvecs V are orthonormal
    // B + lambda I = V * (E + lambda I) * V.T
    // inv(B + lambda I) = V * inv(E + lambda I) * V.T
    // w = V.T * g
    // s(lam) = V * w./(eigvals + lam)
    // ds(lam) = - V * w./((eigvals + lam)**2)
    // \phi(lam) = 1/||s(lam)|| - 1/delta
    // \phi'(lam) = - s(lam).T*ds(lam)/||s(lam)||^3

    double laminit;

    // POSITIVE DEFINITE
    if (mineig > 0) {
        // positive definite
        auto s = slam(0.0, w, eigvals, eigvecs); // s = - self.cB\self.cg_hat
        if (norm(s) <= delta + sqrt(std::numeric_limits<double>::epsilon())) {
            // CASE 0
            spdlog::debug("Interior subproblem solution");
            return { s, "posdef" };
        }
        laminit = 0;
    } else {
        laminit = -mineig;
    }

    // INDEFINITE CASE
    // note that this includes what Nocedal calls the "hard case" but with
    // ||s|| > delta, so the provided formula is not applicable,
    // the respective w should be close to 0 anyways

    if (secular(laminit, w, eigvals, eigvecs, delta) < 0.0) {
        int maxiter = 100;
        try {
            auto r = newton(
              [&w, &eigvals, &eigvecs, &delta](auto x) {
                  return secular(x, w, eigvals, eigvecs, delta);
              },
              [&w, &eigvals, &eigvecs](auto x) {
                  return dsecular(x, w, eigvals, eigvecs);
              },
              laminit,
              1e-12,
              maxiter);
            auto s = slam(r, w, eigvals, eigvecs);
            if (norm(s) <= delta + 1e-12) {
                spdlog::debug("Found boundary subproblem solution via Newton.");
                return { s, "indef" };
            }
        } catch (std::runtime_error const&) {
        }
        try {
            auto xa = laminit;
            auto xb =
              (laminit + std::sqrt(std::numeric_limits<double>::epsilon())) *
              10;
            // search to the right for a change of sign
            while (secular(xb, w, eigvals, eigvecs, delta) < 0 && maxiter > 0) {
                xa = xb;
                xb *= 10;
                maxiter -= 1;
            }
            if (maxiter > 0) {
                auto r = brentq(
                  [&w, &eigvals, &eigvecs, &delta](auto x) {
                      return secular(x, w, eigvals, eigvecs, delta);
                  },
                  xa,
                  xb,
                  1e-12,
                  maxiter);
                auto s = slam(r, w, eigvals, eigvecs);
                if (norm(s) <=
                    delta + std::sqrt(std::numeric_limits<double>::epsilon())) {
                    spdlog::debug(
                      "Found boundary subproblem solution via brentq.");
                    return { s, "indef" };
                }
            }
        } catch (std::runtime_error const&) {
            // may end up here due to ill-conditioning, treat as hard case
        }
    }

    // HARD CASE (gradient is orthogonal to eigenvector to smallest eigenvalue)
    w = blaze::map(w, eigvals, [mineig](auto w, auto eigval) {
        if (eigval - mineig == 0.0)
            return 0.0;
        return w;
    });
    auto s = slam(-mineig, w, eigvals, eigvecs);

    // we know that ||s(lam) + sigma*v_jmin|| = delta, since v_jmin is
    // orthonormal, we can just subtract the difference in norm to get
    // the right length.

    auto sigma = sqrt(std::max(std::pow(delta, 2) - std::pow(norm(s), 2), 0.0));
    s = s + sigma * column(eigvecs, jmin);
    spdlog::debug("Found boundary 2D subproblem solution via hard case.");
    return { s, "hard" };
}

DynamicVector<double>
slam(double const lam,
     DynamicVector<double> const& w,
     DynamicVector<double> const& eigvals,
     DynamicMatrix<double> const& eigvecs)
{
    auto el = eigvals + lam;
    auto c = blaze::map(w, el, [](auto w, auto el) {
        if (el != 0.0)
            return w / el;
        return w;
    });
    return eigvecs * c;
}

DynamicVector<double>
dslam(double const lam,
      DynamicVector<double> const& w,
      DynamicVector<double> const& eigvals,
      DynamicMatrix<double> const& eigvecs)
{
    auto c = w;
    auto el = eigvals + lam;

    c = blaze::map(c, el, [](auto c, auto el) {
        if (el != 0.0)
            return -c / pow(el, 2);
        if (c != 0.0)
            return std::numeric_limits<double>::infinity();
        return c;
    });

    return eigvecs * c;
}

double
secular(const double lam,
        DynamicVector<double> const& w,
        DynamicVector<double> const& eigvals,
        DynamicMatrix<double> const& eigvecs,
        const double delta)
{
    if (lam < -min(eigvals)) {
        // safeguard to implement boundary
        return std::numeric_limits<double>::infinity();
    }

    auto s = slam(lam, w, eigvals, eigvecs);
    auto sn = norm(s);

    if (sn > 0)
        return 1 / sn - 1 / delta;

    return std::numeric_limits<double>::infinity();
}

double
dsecular(double const lam,
         DynamicVector<double> const& w,
         DynamicVector<double> const& eigvals,
         DynamicMatrix<double> const& eigvecs)
{
    auto s = slam(lam, w, eigvals, eigvecs);
    auto ds = dslam(lam, w, eigvals, eigvecs);
    auto sn = norm(s);
    if (sn > 0.0)
        return -trans(s) * ds / std::pow(sn, 3);
    return std::numeric_limits<double>::infinity();
}

} // namespace fides
