/**
 * Hessian Update Strategies
 * -------------------------
 *
 * This module provides various generic Hessian approximation strategies that
 * can be employed when the calculating the exact Hessian or an approximation
 * is computationally too demanding.
 */

#ifndef FIDES_HESSIAN_APPROXIMATION_HPP
#define FIDES_HESSIAN_APPROXIMATION_HPP

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

using blaze::DynamicMatrix;
using blaze::DynamicVector;

namespace fides {

/**
 * @brief Abstract class from which Hessian update strategies should be derived.
 */
class HessianApproximation {
  public:
    HessianApproximation() = default;

    /**
     * @brief Create a Hessian update strategy instance
     * @param hess_init Initial guess for the Hessian. If empty Identity matrix
     * will be used.
     */
    HessianApproximation(DynamicMatrix<double> const &hess_init) {
        if(!blaze::isEmpty(hess_init_))
            set_init(hess_init);
    }

    virtual void set_init(DynamicMatrix<double> const &hess_init);
    /**
     * @brief Initializes this approximation instance and checks the
     * dimensionality.
     * @param dim Dimension of optimization variables.
     */
    virtual void init_mat(const int dim);

    /**
     * @brief Update Hessian approximation
     * @param s
     * @param y
     */
    virtual void update(DynamicVector<double> const &s,
                        DynamicVector<double> const &y) = 0;

    /**
     * @brief Get current Hessian approximation
     * @return Current Hessian approximation
     */
    virtual DynamicMatrix<double> const &get_mat() const { return hess_; }

  protected:
    DynamicMatrix<double> hess_;

  private:
    DynamicMatrix<double> hess_init_;
};

/**
 * @brief Symmetric Rank-1 update strategy.
 *
 * This updating strategy may yield indefinite hessian approximations.
 */
class SR1 : public HessianApproximation {
  public:
    void update(DynamicVector<double> const &s,
                DynamicVector<double> const &y) override;
};

/**
 * @brief Broyden-Fletcher-Goldfarb-Shanno update strategy.
 *
 * This is a rank-2 update strategy that always yields positive semi-definite
 * Hessian approximations.
 */
class BFGS : public HessianApproximation {
  public:
    using HessianApproximation::HessianApproximation;
    void update(DynamicVector<double> const &s,
                DynamicVector<double> const &y) override;
};

/**
 * @brief Davidon-Fletcher-Powell update strategy.
 *
 * This is a rank-2 update strategy that always yields positive semi-definite
 * Hessian approximations. It usually does not perform as well as the BFGS
 * strategy, but included for the sake of completeness.
 */
class DFP : public HessianApproximation {
  public:
    void update(DynamicVector<double> const &s,
                DynamicVector<double> const &y) override;
};

class HybridUpdate : public HessianApproximation {
  public:
    /**
     * @brief Create a Hybrid Hessian update strategy which is generated from
     * the start but only applied after a certain iteration, while Hessian
     * computed by the objective function is used until then.
     *
     * @param happ Hessian Update Strategy (default: BFGS)
     * @param hess_init Iteration after which this approximation is used
     * (default: 2*dim)
     * @param switch_iteration Initial guess for the Hessian. (default: eye)
     */
    HybridUpdate(std::unique_ptr<HessianApproximation> happ,
                 DynamicMatrix<double> const &hess_init,
                 const int switch_iteration)
        : HessianApproximation(hess_init), switch_iteration_(switch_iteration),
          hessian_update_(std::move(happ))

    {
        if (!hessian_update_)
            hessian_update_ = std::make_unique<BFGS>();
    }

    /**
     * @brief HybridUpdate
     * @param happ
     * @param init_with_hess Whether the hybrid update strategy should be
     * initialized according to the user-provided objective function
     * @param switch_iteration
     */
    HybridUpdate(std::unique_ptr<HessianApproximation> happ,
                 bool init_with_hess,
                 const int switch_iteration)
        : switch_iteration_(switch_iteration), init_with_hess_(init_with_hess)
          ,hessian_update_(std::move(happ))

    {
        if (!hessian_update_)
            hessian_update_ = std::make_unique<BFGS>();
    }

    void set_init(DynamicMatrix<double> const &hess_init) override {
        hessian_update_->set_init(hess_init);
    }

    void init_mat(const int dim) override {
        if (switch_iteration_ < 0) {
            switch_iteration_ = 2 * dim;
        }
        hessian_update_->init_mat(dim);
    }

    DynamicMatrix<double> const &get_mat() const override {
        return hessian_update_->get_mat();
    }

    void update(DynamicVector<double> const &s,
                DynamicVector<double> const &y) override {
        hessian_update_->update(s, y);
    }

    int switch_iteration_ {-1};

    bool init_with_hess_ {false};

  private:
    std::unique_ptr<HessianApproximation> hessian_update_;
};

} // namespace fides

#endif // FIDES_HESSIAN_APPROXIMATION_HPP
