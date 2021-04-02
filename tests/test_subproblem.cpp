#include "gtest/gtest.h"

#include <blaze/math/StaticMatrix.h>
#include <exception>

#include "misc.hpp"
#include <fides/steps.hpp>
#include <fides/subproblem.hpp>

namespace fides {
namespace {

auto quad(DynamicVector<double> s, DynamicMatrix<double> B,
          DynamicVector<double> g) {
    return 0.5 * trans(s) * (B * s) + trans(s) * g;
}

bool is_local_quad_min(DynamicVector<double> s, DynamicMatrix<double> B,
                       DynamicVector<double> g) {
    // make local perturbations to verify s is a local minimum of quad(s, B, g)

    DynamicVector<std::complex<double>> eigvals_c(B.rows());
    DynamicMatrix<std::complex<double>> eigvecs_c(B.rows(), B.rows());
    eigen(B, eigvals_c, eigvecs_c);
    auto eigvecs = trans(real(eigvecs_c));

    auto quad_unperturbed = quad(s, B, g);

    for (auto eps : {1e-2, -1e-2}) {
        for (std::size_t iv = 0; iv < eigvecs.columns(); ++iv) {
            auto s_perturbed = s + eps * blaze::column(eigvecs, iv);
            auto quad_perturbed = quad(s_perturbed, B, g);
            if (quad_perturbed < quad_unperturbed)
                return false;
        }
    }
    return true;
}

bool is_bound_quad_min(DynamicVector<double> s, DynamicMatrix<double> B,
                       DynamicVector<double> g) {
    // make local rotations to verify that s is a local minimum of quad(s, B, g)
    // on the sphere of radius ||s||

    auto quad_unperturbed = quad(s, B, g);
    const auto pi = acos(-1);
    const double theta = pi / 2.0;

    for (auto eps : {1e-2, -1e-2}) {
        for (std::size_t iv = 0; iv < B.columns(); ++iv) {
            // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
            // v_rot = (cos_theta) * v + sin_theta * (e x v) + (1 - cos_theta) *
            // (e*v)
            // * e
            auto cur_eye = DynamicVector<double>(3UL, 0.0);
            cur_eye[iv] = 1.0;
            auto rotation_axis = blaze::normalize(theta * eps * cur_eye);
            auto s_rotated =
                std::cos(theta) * s +
                std::sin(theta) * blaze::cross(rotation_axis, s) +
                (1 - std::cos(theta)) * (rotation_axis * s) * rotation_axis;

            auto quad_perturbed = quad(s_rotated, B, g);
            if (quad_perturbed < quad_unperturbed)
                return false;
        }
    }
    return true;
}

class Subproblem : public ::testing::Test {
  protected:
    void SetUp() override {
        B_ = DynamicMatrix<double>{
            {1.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 5.0}};

        g_ = DynamicVector<double>{1.0, 1.0, 1.0};
    }

    // void TearDown() override {}

    DynamicMatrix<double> B_;
    DynamicVector<double> g_;
};

TEST_F(Subproblem, test_convex_subproblem) {
    double delta = 1.151;
    auto [s, case_] = solve_nd_trust_region_subproblem(B_, g_, delta);
    auto w = real(eigen(B_));
    ASSERT_GT(min(w), 0.0);
    ASSERT_LT(blaze::norm(s), delta);
    ASSERT_EQ(case_, "posdef");
    ASSERT_TRUE(is_local_quad_min(s, B_, g_));
};

TEST_F(Subproblem, test_nonconvex_subproblem) {
    B_(0, 0) = -1.0;
    auto delta = 1.0;
    auto [s, case_] = solve_nd_trust_region_subproblem(B_, g_, delta);

    auto w = real(eigen(B_));
    ASSERT_LT(min(w), 0.0);
    ASSERT_NEAR(norm(s), delta, 1e-6);
    ASSERT_EQ(case_, "indef");
    ASSERT_TRUE(is_bound_quad_min(s, B_, g_));

    auto snorm = normalize(s);
    for (auto alpha : {0.0, 0.5, -0.5}) {
        auto sol =
            solve_1d_trust_region_subproblem(B_, g_, snorm, delta, alpha * s);
        ASSERT_NEAR(sol[0], (1 - alpha) * delta, 1e-8);
    }
};

TEST_F(Subproblem, test_nonconvex_subproblem_eigvals) {
    auto minevs = blaze::logspace(50U, -1, -50);
    for (auto minev : minevs) {
        B_(0, 0) = -minev;
        auto delta = 1.0;
        auto [s, case_] = solve_nd_trust_region_subproblem(B_, g_, delta);

        auto w = real(eigen(B_));
        ASSERT_LT(min(w), 0.0);
        ASSERT_NEAR(norm(s), delta, 1e-6);
        ASSERT_TRUE(is_bound_quad_min(s, B_, g_));

        auto snorm = normalize(s);
        for (auto alpha : {0.0, 0.5, -0.5}) {
            auto sol = solve_1d_trust_region_subproblem(B_, g_, snorm, delta,
                                                        alpha * s);
            ASSERT_NEAR(sol[0], (1 - alpha) * delta, 1e-8);
        }
    }
}

TEST_F(Subproblem, test_hard_indef_subproblem) {
    B_(0, 0) = -1.0;
    g_[0] = 0.0;
    auto delta = 0.1;
    auto [s, case_] = solve_nd_trust_region_subproblem(B_, g_, delta);

    auto w = real(eigen(B_));
    ASSERT_LT(min(w), 0.0);
    ASSERT_NEAR(norm(s), delta, 1e-6);
    ASSERT_EQ(case_, "indef");
    ASSERT_TRUE(is_bound_quad_min(s, B_, g_));

    auto snorm = normalize(s);
    for (auto alpha : {0.0, 0.5, -0.5}) {
        auto sol =
            solve_1d_trust_region_subproblem(B_, g_, snorm, delta, alpha * s);
        ASSERT_NEAR(sol[0], (1 - alpha) * delta, 1e-8);
    }
}

TEST_F(Subproblem, test_hard_hard_subproblem) {
    B_(0, 0) = -1.0;
    g_[0] = 0.0;
    auto delta = 0.5;
    auto [s, case_] = solve_nd_trust_region_subproblem(B_, g_, delta);

    auto w = real(eigen(B_));
    ASSERT_LT(min(w), 0.0);
    ASSERT_NEAR(norm(s), delta, 1e-6);
    ASSERT_EQ(case_, "hard");
    ASSERT_TRUE(is_bound_quad_min(s, B_, g_));

    auto snorm = normalize(s);
    for (auto alpha : {0.0, 0.5, -0.5}) {
        auto sol =
            solve_1d_trust_region_subproblem(B_, g_, snorm, delta, alpha * s);
        ASSERT_NEAR(sol[0], (1 - alpha) * delta, 1e-8);
    }
}

TEST(newton, test_newton) {
    auto f = [](double x) { return pow(x, 3) - 1.0; };

    auto fprime = [](double x) { return 3 * pow(x, 2); };

    auto root = newton(f, fprime, 1.5, 1.48e-8, 50);
    EXPECT_DOUBLE_EQ(root, 1.0);
}

TEST(brentq, test_brentq) {
    auto f = [](double x) { return pow(x, 2) - 1.0; };

    auto root = brentq(f, -2.0, 0.0, 2e-12, 100);
    EXPECT_DOUBLE_EQ(root, -1.0);

    root = brentq(f, 0.0, 2.0, 2e-12, 100);
    EXPECT_DOUBLE_EQ(root, 1.0);
}

} // namespace
} // namespace fides
