#include "fides/steps.hpp"

#include <gsl/gsl-lite.hpp>

#include <blaze/Math.h>

#include <spdlog/spdlog.h>

#include <exception>
#include <iostream>

extern "C"
{
    using blaze::blas_int_t;

    /**
     * @brief LAPACK dgelsd_ prototype.
     */
    void dgelsd_(blas_int_t* m,
                 blas_int_t* n,
                 blas_int_t* nrhs,
                 double* A,
                 blas_int_t* lda,
                 double* b,
                 blas_int_t* ldb,
                 double* s,
                 double* rcond,
                 blas_int_t* rank,
                 double* work,
                 blas_int_t* lwork,
                 blas_int_t* iwork,
                 blas_int_t* info);

} // extern "C"

namespace fides {

DynamicVector<double>
dgelsd(DynamicMatrix<double>& A, const DynamicVector<double> b)
{
    // currently restricted to square
    Expects(A.rows() == A.columns());
    Expects(A.rows() == b.size());

    // col-major for LAPACK
    Expects(A.storageOrder == blaze::rowMajor);
    DynamicMatrix<double, blaze::columnMajor> A_(A);

    // copy, because output will be overwritten
    DynamicVector<double> res(b);

    blas_int_t m = A_.rows();
    blas_int_t n = A_.columns();
    blas_int_t nrhs = 1;
    blas_int_t lda = m;
    blas_int_t ldb = A_.rows();
    DynamicVector<double> s(std::min(A.rows(), A.columns()));
    double rcond = -1.0;
    int rank;
    std::vector<double> work(1);
    blas_int_t lwork = -1;
    int minmn = std::min(m, n);
    int smlsiz = 25;
    int nlvl = std::max(0, int(log2(minmn / (smlsiz + 1))) + 1);

    std::vector<blas_int_t> iwork(std::max(1, 3 * minmn * nlvl + 11 * minmn));
    blas_int_t info;

    // get work size
    dgelsd_(&m,
            &n,
            &nrhs,
            A_.data(),
            &lda,
            res.data(),
            &ldb,
            s.data(),
            &rcond,
            &rank,
            work.data(),
            &lwork,
            iwork.data(),
            &info);

    lwork = work[0];
    work.resize(lwork);
    dgelsd_(&m,
            &n,
            &nrhs,
            A_.data(),
            &lda,
            res.data(),
            &ldb,
            s.data(),
            &rcond,
            &rank,
            work.data(),
            &lwork,
            iwork.data(),
            &info);

    if (info)
        throw std::runtime_error("dgelsd: " + std::to_string(info));

    return res;
}

double
quadratic_form(const DynamicMatrix<double>& Q,
               const DynamicVector<double>& p,
               const DynamicVector<double>& x)
{
    return 0.5 * (trans(x) * Q) * x + trans(p) * x;
}

Step::Step(DynamicVector<double> const& x,
           DynamicVector<double> const& sg,
           DynamicMatrix<double> const& hess,
           CompressedMatrix<double> const& scaling,
           CompressedMatrix<double> const& g_dscaling,
           const double delta,
           const double theta,
           DynamicVector<double> const& lb,
           DynamicVector<double> const& ub)
  : x_(x)
  , sg_(sg)
  , scaling_(scaling)
  , delta_(delta)
  , theta_(theta)
  , lb_(lb)
  , ub_(ub)

{
    br_ = DynamicVector<double>(sg.size(), 1.0);
    shess_ = scaling_ * hess * scaling_ + g_dscaling;
    s0_ = blaze::zero<double>(sg.size());
    ss0_ = blaze::zero<double>(sg.size());
}

void
Step::step_back()
{
    // create copies of the calculated step
    og_s_ = s_;
    og_ss_ = ss_;
    og_sc_ = sc_;

    br_ =
      DynamicVector<double>(s_.size(), std::numeric_limits<double>::infinity());
    if (!blaze::isZero<blaze::strict>(s_)) {
        br_ = blaze::map(
          br_, x_, s_, lb_, ub_, [](auto br, auto x, auto s, auto lb, auto ub) {
              if (s != 0.0) {
                  return std::max((ub - x) / s, (lb - x) / s);
              }
              return br;
          });
    }
    minbr_ = blaze::min(br_);
    iminbr_.clear();
    for (size_t i = 0; i < br_.size(); ++i)
        if (br_[i] == minbr_)
            iminbr_.push_back(i);

    // compute the minimum of the step
    alpha_ = std::min(1.0, theta_ * minbr_);
    s_ *= alpha_;
    sc_ *= alpha_;
    ss_ *= alpha_;
}

void
Step::reduce_to_subspace()
{
    chess_ = trans(subspace_) * shess_ * subspace_;
    cg_ = trans(subspace_) * sg_;
}

void
Step::compute_step()
{
    if (subspace_.columns() == 0) {
        sc_ = DynamicVector<double>(); //? empty((0, 0))
        ss_ = blaze::zero<double>(ss0_.size());
        s_ = blaze::zero<double>(s0_.size());
        return;
    }

    if (subspace_.columns() > 1) {
        std::string _;
        auto delta =
          sqrt(std::max(std::pow(delta_, 2) - std::pow(norm(ss0_), 2), 0.0));
        std::tie(sc_, _) = solve_nd_trust_region_subproblem(chess_, cg_, delta);
    } else {
        sc_ = solve_1d_trust_region_subproblem(
          shess_, sg_, blaze::column(subspace_, 0U), delta_, ss0_);
    }
    ss_ = subspace_ * real(sc_);
    s_ = scaling_ * ss_;
}

void
Step::calculate()
{
    reduce_to_subspace();
    compute_step();
    step_back();
    qpval_ = quadratic_form(shess_, sg_, ss_ + ss0_);
}

TRStep2D::TRStep2D(const DynamicVector<double>& x,
                   const DynamicVector<double>& sg,
                   const DynamicMatrix<double>& hess,
                   const CompressedMatrix<double>& scaling,
                   const CompressedMatrix<double>& g_dscaling,
                   const double delta,
                   const double theta,
                   const DynamicVector<double>& lb,
                   const DynamicVector<double>& ub)
  : Step(x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub)
{
    type = "tr2d";
    auto n = sg.size();

    DynamicVector<double> s_newt(sg.size());
    s_newt = -dgelsd(shess_, sg);
    auto posdef = blaze::dot(s_newt, shess_ * s_newt) > 0.0;
    s_newt = normalize(s_newt);
    DynamicVector<double> s_grad;
    if (n > 1) {
        if (!posdef) {
            // in this case we are in Case 2 of Fig 12 in [Coleman-Li1994]
            spdlog::debug("Newton direction did not have negative curvature "
                          "adding scaling * sign(sg) to 2D subspace.");

            s_grad = blaze::map(scaling * sign(sg), sg, [](auto tmp, auto sg) {
                return tmp + (sg == 0.0);
            });

        } else {
            s_grad = sg;
        }
        // orthonormalize, this ensures that S.T.dot(S) = I and we
        // can use S/S.T for transformation
        s_grad = s_grad - s_newt * blaze::dot(s_newt, s_grad);
        s_grad = normalize(s_grad);

        // if non-zero, add s_grad to subspace
        if (!isZero(s_grad)) {
            subspace_ = blaze::zero<double>(s_newt.size(), 2U);
            blaze::column(subspace_, 0U) = s_newt;
            blaze::column(subspace_, 1U) = s_grad;
            return;
        }
        spdlog::debug("Singular subspace, continuing with 1D subspace.");
    }
    subspace_ = blaze::zero<double>(s_newt.size(), 1U);
    blaze::column(subspace_, 0U) = s_newt;
}

RefinedStep::RefinedStep(const DynamicVector<double>& x,
                         const DynamicVector<double>& sg,
                         const DynamicMatrix<double>& hess,
                         const CompressedMatrix<double>& scaling,
                         const CompressedMatrix<double>& g_dscaling,
                         const double delta,
                         const double theta,
                         const DynamicVector<double>& lb,
                         const DynamicVector<double>& ub,
                         const Step& step)
  : Step(x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub)
{
    type = "ref";

    subspace_ = blaze::DynamicMatrix<double>(sg.size(), 1U);
    blaze::column(subspace_, 0U) = blaze::normalize(sg);
    // TODO RefinedStep
    //         self.constraints = [
    //             NonlinearConstraint(
    //                 fun=lambda xs: (norm(xs) - delta) * np.ones((1,)),
    //                 jac=lambda xs: np.expand_dims(xs, 1).T / norm(xs),
    //                 lb=np.zeros((1,)),
    //                 ub=np.ones((1,)) * np.inf,
    //             )
    //         ]
    //         self.guess = step.ss + step.ss0
    //         self.bounds = Bounds(
    //             step.theta * (lb - x) / scaling.diagonal(),
    //             step.theta * (ub - x) / scaling.diagonal()
    //         )

    // NOT YET IMPLEMENTED
    std::terminate();

    reflection_count_ = step.reflection_count_;
    truncation_count_ = step.truncation_count_;
}

void
RefinedStep::calculate()
{
    // TODO RefinedStep
    //         res = minimize(fun=lambda s: quadratic_form(self.shess,
    //         self.sg, s),
    //                        jac=lambda s: self.shess.dot(s) + self.sg,
    //                        hess=lambda s: self.shess,
    //                        x0=self.guess,
    //                        method='trust-constr',
    //                        bounds=self.bounds,
    //                        constraints=self.constraints,
    //                        options={'verbose': 0, 'maxiter': 10})
    //         self.ss = res.x
    //         self.s = self.scaling.dot(res.x)

    // NOT YET IMPLEMENTED
    std::terminate();

    sc_ = ss_;
    step_back();
    qpval_ = quadratic_form(shess_, sg_, ss0_);
}

GradientStep::GradientStep(const DynamicVector<double>& x,
                           const DynamicVector<double>& sg,
                           const DynamicMatrix<double>& hess,
                           const CompressedMatrix<double>& scaling,
                           const CompressedMatrix<double>& g_dscaling,
                           const double delta,
                           const double theta,
                           const DynamicVector<double>& lb,
                           const DynamicVector<double>& ub)
  : Step(x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub)
{
    type = "g";
    subspace_ = blaze::DynamicMatrix<double>(sg.size(), 1U);
    blaze::column(subspace_, 0U) = blaze::normalize(sg);
}

TRStepTruncated::TRStepTruncated(DynamicVector<double> const& x,
                                 DynamicVector<double> const& sg,
                                 DynamicMatrix<double> const& hess,
                                 CompressedMatrix<double> const& scaling,
                                 CompressedMatrix<double> const& g_dscaling,
                                 const double delta,
                                 const double theta,
                                 DynamicVector<double> const& lb,
                                 DynamicVector<double> const& ub,
                                 Step const& step)
  : Step(x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub)
{
    type = "trt";

    s0_ = step.s0_;
    ss0_ = step.ss0_;
    iminbr_ = step.iminbr_;
    blaze::elements(s0_, iminbr_) += step.theta_ *
                                     blaze::elements(step.br_, iminbr_) *
                                     blaze::elements(step.og_s_, iminbr_);
    blaze::elements(ss0_, iminbr_) += step.theta_ *
                                      blaze::elements(step.br_, iminbr_) *
                                      blaze::elements(step.og_ss_, iminbr_);

    // update x and at breakpoint
    x_ = x + s0_;

    subspace_ = step.subspace_;
    blaze::rows(subspace_, iminbr_) = 0.0;

    // reduce and normalize subspace
    std::vector<size_t> non_zero_cols;
    for (size_t i = 0; i < subspace_.columns(); ++i)
        if (!blaze::isZero<blaze::strict>(blaze::column(subspace_, i)))
            non_zero_cols.push_back(i);
    auto tmp_subspace =
      DynamicMatrix<double>(subspace_.rows(), non_zero_cols.size());
    for (size_t i = 0; i < tmp_subspace.columns(); ++i)
        blaze::column(tmp_subspace, i) =
          blaze::normalize(blaze::column(subspace_, non_zero_cols.at(i)));
    subspace_ = tmp_subspace;

    truncation_count_ = step.truncation_count_ + iminbr_.size();
}

TRStepReflected::TRStepReflected(DynamicVector<double> const& x,
                                 DynamicVector<double> const& sg,
                                 DynamicMatrix<double> const& hess,
                                 CompressedMatrix<double> const& scaling,
                                 CompressedMatrix<double> const& g_dscaling,
                                 const double delta,
                                 const double theta,
                                 DynamicVector<double> const& lb,
                                 DynamicVector<double> const& ub,
                                 Step const& step)
  : Step(x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub)
{
    type = "trr";

    auto alpha = std::min(step.minbr_, 1.0);
    s0_ = alpha * step.og_s_ + step.s0_;
    ss0_ = alpha * step.og_ss_ + step.ss0_;

    // update x and at breakpoint
    x_ = x + s0_;

    // reflect the transformed step at the boundary
    auto nss = step.og_ss_;
    blaze::elements(nss, step.iminbr_) *= -1.0;
    subspace_ = blaze::DynamicMatrix<double>(nss.size(), 1U);
    blaze::column(subspace_, 0U) = blaze::normalize(nss);
    reflection_count_ = step.reflection_count_ + 1;
}

TRStepFull::TRStepFull(const DynamicVector<double>& x,
                       const DynamicVector<double>& sg,
                       const DynamicMatrix<double>& hess,
                       const CompressedMatrix<double>& scaling,
                       const CompressedMatrix<double>& g_dscaling,
                       const double delta,
                       const double theta,
                       const DynamicVector<double>& lb,
                       const DynamicVector<double>& ub)
  : Step(x, sg, hess, scaling, g_dscaling, delta, theta, lb, ub)
{
    type = "trnd";
    subspace_ = blaze::IdentityMatrix<double>(hess.rows());
}

} // namespace fides
