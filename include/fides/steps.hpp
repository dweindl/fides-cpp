/**
 * Trust Region Step Calculation
 * -----------------------------
 * This module provides the machinery to calculate different trust-region(
 * -reflective) step proposals
 */

#ifndef FIDES_STEPS_HPP
#define FIDES_STEPS_HPP

#include <fides/subproblem.hpp>

#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;

namespace fides {

/**
 * @brief Computes the quadratic form \f[ x^TQx + x^Tp \f]
 * @param Q Matrix
 * @param p Vector
 * @param x Input
 * @return Value of form
 */
double
quadratic_form(DynamicMatrix<double> const& Q,
               DynamicVector<double> const& p,
               DynamicVector<double> const& x);

/**
 * @brief Base class for the computation of a proposal step
 */
class Step
{
  public:
    /**
     * @brief Construct a Step
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     */
    Step(DynamicVector<double> const& x,
         DynamicVector<double> const& sg,
         DynamicMatrix<double> const& hess,
         CompressedMatrix<double> const& scaling,
         CompressedMatrix<double> const& g_dscaling,
         const double delta,
         const double theta,
         DynamicVector<double> const& lb,
         DynamicVector<double> const& ub);

    /**
     * @brief Truncates the step based on the distance of the current point to
     * the boundary.
     */
    void step_back();

    /**
     * @brief Project the matrix shess and the vector sg to the subspace
     */
    void reduce_to_subspace();

    /**
     * @brief Compute the step as solution to the trust region subproblem.
     *
     * Special code is used for the special case 1-dimensional subspace case.
     */
    void compute_step();

    /**
     * @brief Calculates step and the expected objective function value
     * according to the quadratic approximation.
     */
    virtual void calculate();

    /** Identifier that allows identification of subclasses */
    std::string type{ "step" };

    /** Current state of optimization variables */
    DynamicVector<double> x_;

    /** Proposed step */
    DynamicVector<double> s_;

    /**
     * Coefficients in the 1D/2D subspace that defines the affine transformed
     * step ss: `ss = subspace * sc`
     */
    DynamicVector<double> sc_;

    /** Affine transformed step: `s = scaling * ss` */
    DynamicVector<double> ss_;

    /** `s` without step back */
    DynamicVector<double> og_s_;

    /** `sc` without step back */
    DynamicVector<double> og_sc_;

    /** `ss` without step back */
    DynamicVector<double> og_ss_;

    /** Rescaled gradient `scaling * g` */
    DynamicVector<double> sg_;

    /**
     * Quantifies the distance to the boundary normalized by the proposed step,
     * this indicates the fraction of the step that would put the respective
     * variable at the boundary. This is defined in [Coleman-Li1996] (3.1).
     */
    DynamicVector<double> br_;

    /** Maximal fraction of step s that can be taken to reach first breakpoint
     */
    double minbr_{ 1.0 };

    /**
     * Indices of x that specify the variable that will hit the breakpoint if
     * a step minbr * s is taken
     */
    std::vector<size_t> iminbr_;

    /**  */
    CompressedMatrix<double> scaling_;

    /** Trust region radius in the transformed space defined by scaling matrix
     */
    double delta_;

    /** Controls step back, fraction of step to take if full step would reach
     * breakpoint
     */
    double theta_;

    /** Lower boundaries for x */
    DynamicVector<double> lb_;

    /** Upper boundaries for x */
    DynamicVector<double> ub_;

    /**  */
    double alpha_{ 1.0 };

    /** Value of the quadratic subproblem for the proposed step */
    double qpval_{ 1.0 };

    /**  */
    DynamicVector<double> s0_;

    /**  */
    DynamicVector<double> ss0_;

    /** Number of reflections that were applied to obtain this step */
    int reflection_count_{ 0 };

    /** Number of reflections that were applied to obtain this step */
    int truncation_count_{ 0 };

    /** Projection of the g_hat to the subspace */
    DynamicVector<double> cg_;

    /** Projection of the B to the subspace */
    DynamicMatrix<double> chess_;

    /**  */
    DynamicMatrix<double> subspace_;

    /** Matrix of the full quadratic problem */
    DynamicMatrix<double> shess_;
};

/**
 * @brief The TRStepFull class provides the machinery to compute an exact
 * solution of the trust region subproblem.
 */
class TRStepFull : public Step
{
  public:
    /**
     * @brief Construct a TRStepFull
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     */
    TRStepFull(DynamicVector<double> const& x,
               DynamicVector<double> const& sg,
               DynamicMatrix<double> const& hess,
               CompressedMatrix<double> const& scaling,
               CompressedMatrix<double> const& g_dscaling,
               const double delta,
               const double theta,
               DynamicVector<double> const& lb,
               DynamicVector<double> const& ub);
};

/**
 * @brief The TRStep2D class provides the machinery to compute an approximate
 * solution of the trust region subproblem according to a 2D subproblem.
 */
class TRStep2D : public Step
{
  public:
    /**
     * @brief Construct a TRStep2D
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     */
    TRStep2D(DynamicVector<double> const& x,
             DynamicVector<double> const& sg,
             DynamicMatrix<double> const& hess,
             CompressedMatrix<double> const& scaling,
             CompressedMatrix<double> const& g_dscaling,
             const double delta,
             const double theta,
             DynamicVector<double> const& lb,
             DynamicVector<double> const& ub);
};

/**
 * @brief The TRStepReflected class provides the machinery to compute a
 * reflected step based on trust region subproblem solution that hit the
 * boundaries.
 */
class TRStepReflected : public Step
{
  public:
    /**
     * @brief Construct a TRStepReflected
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     * @param step Trust-region step that is reflected
     */
    TRStepReflected(DynamicVector<double> const& x,
                    DynamicVector<double> const& sg,
                    DynamicMatrix<double> const& hess,
                    CompressedMatrix<double> const& scaling,
                    CompressedMatrix<double> const& g_dscaling,
                    const double delta,
                    const double theta,
                    DynamicVector<double> const& lb,
                    DynamicVector<double> const& ub,
                    Step const& step);
};

/**
 * @brief The TRStepTruncated class provides the machinery to compute a reduced
 * step based on trust region subproblem solution that hit the boundaries.
 */
class TRStepTruncated : public Step
{
  public:
    /**
     * @brief Construct a TRStepReflected
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     * @param step Trust-region step that is reduced
     */
    TRStepTruncated(DynamicVector<double> const& x,
                    DynamicVector<double> const& sg,
                    DynamicMatrix<double> const& hess,
                    CompressedMatrix<double> const& scaling,
                    CompressedMatrix<double> const& g_dscaling,
                    const double delta,
                    const double theta,
                    DynamicVector<double> const& lb,
                    DynamicVector<double> const& ub,
                    Step const& step);
};

/**
 * @brief The GradientStep class provides the machinery to compute a gradient
 * step.
 */
class GradientStep : public Step
{
  public:
    /**
     * @brief Construct a GradientStep
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     */
    GradientStep(DynamicVector<double> const& x,
                 DynamicVector<double> const& sg,
                 DynamicMatrix<double> const& hess,
                 CompressedMatrix<double> const& scaling,
                 CompressedMatrix<double> const& g_dscaling,
                 const double delta,
                 const double theta,
                 DynamicVector<double> const& lb,
                 DynamicVector<double> const& ub);
};

/**
 * @brief The RefinedStep class provides the machinery to refine a step based
 * on interior point optimization.
 */
class RefinedStep : public Step
{
  public:
    /**
     * @brief Construct a RefinedStep
     * @param x Reference point
     * @param sg Gradient in rescaled coordinates
     * @param hess Hessian in unscaled coordinates
     * @param scaling Matrix that defines scaling transformation
     * @param g_dscaling Unscaled gradient multiplied by derivative of scaling
     * transformation
     * @param delta Trust region Radius in scaled coordinates
     * @param theta Stepback parameter that controls how close steps are allowed
     * to get to the boundary
     * @param lb Lower boundary
     * @param ub Upper boundary
     * @param step Trust-region step that is refined
     */
    RefinedStep(DynamicVector<double> const& x,
                DynamicVector<double> const& sg,
                DynamicMatrix<double> const& hess,
                CompressedMatrix<double> const& scaling,
                CompressedMatrix<double> const& g_dscaling,
                const double delta,
                const double theta,
                DynamicVector<double> const& lb,
                DynamicVector<double> const& ub,
                Step const& step);

    virtual void calculate() override;
};

/**
 * @brief LAPACK's dgelsd for blaze types.
 *
 * Computes the minimum-norm solution to a real linear least squares problem.
 * (http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga94bd4a63a6dacf523e25ff617719f752.html)
 *
 * @param A
 * @param b
 * @return
 */
DynamicVector<double>
dgelsd(DynamicMatrix<double>& A, const DynamicVector<double> b);

} // namespace fides

#endif // FIDES_STEPS_HPP
