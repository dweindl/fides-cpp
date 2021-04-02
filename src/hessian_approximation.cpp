#include "fides/hessian_approximation.hpp"

#include <exception>

namespace fides {

void HessianApproximation::set_init(const DynamicMatrix<double> &hess_init) {
    if (hess_init.rows() != hess_init.columns()) {
        throw std::domain_error("hess_init is not square.");
    }
    hess_init_ = hess_init;
}

void HessianApproximation::init_mat(const int dim) {
    if (!blaze::isEmpty(hess_init_)) {
        if (hess_init_.rows() != static_cast<std::size_t>(dim)) {
            throw std::domain_error(
                "Initial approximation has wrong dimension.");
        }
        hess_ = hess_init_;
    } else {
        hess_ = blaze::IdentityMatrix<double>(dim);
    }
}

void SR1::update(const DynamicVector<double> &s, const DynamicVector<double> &y) {
    auto z = y - hess_ * s;
    auto d = trans(z) * s;

    // [NocedalWright2006] (6.26) reject if update degenerate
    if (std::abs(d) >= 1e-8 * norm(s) * norm(z))
        hess_ += outer(z, trans(z)) / d;
}

void BFGS::update(const DynamicVector<double> &s, const DynamicVector<double> &y) {
    auto b = trans(y) * s;
    if (b < 0.0) {
        return;
    }
    auto z = hess_ * s;
    auto a = trans(s) * z;
    hess_ += -outer(z, trans(z)) / a + outer(y, trans(y)) / b;
}

void DFP::update(const DynamicVector<double> &s, const DynamicVector<double> &y) {
    auto curv = trans(y) * s;
    if (curv <= 0.0)
        return;

    auto mat1 = blaze::IdentityMatrix<double>(hess_.rows()) -
                outer(y, trans(s)) / curv;
    auto mat2 = blaze::IdentityMatrix<double>(hess_.rows()) -
                outer(s, trans(y)) / curv;
    hess_ = mat1 * hess_ * mat2 + outer(y, trans(y)) / curv;
}

} // namespace fides
