#include "gtest/gtest.h"

#include <exception>

#include <fides/minimize.hpp>

#include "misc.hpp"

namespace fides {
namespace {

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

double
rosen(DynamicVector<double> x)
{
    return 100 * std::pow(x[1] - std::pow(x[0], 2), 2) + std::pow(1 - x[0], 2);
}

cost_fun_ret_t
rosengrad(DynamicVector<double> x)
{

    auto f = rosen(x);
    auto g = DynamicVector<double>{ -400.0 * (x[1] - std::pow(x[0], 2)) * x[0] -
                                      2.0 * (1.0 - x[0]),
                                    200.0 * (x[1] - std::pow(x[0], 2)) };
    auto h = DynamicMatrix<double>();

    return std::make_tuple(f, g, h);
}

cost_fun_ret_t
rosenboth(DynamicVector<double> x)
{
    auto [f, g, _] = rosengrad(x);
    auto h = DynamicMatrix<double>{ { 1200 * std::pow(x[0], 2) - 400 * x[1] + 2,
                                      -400 * x[0] },
                                    { -400 * x[0], 200 } };
    return std::make_tuple(f, g, h);
}

cost_fun_ret_t
rosenrandomfail(DynamicVector<double> x)
{
    auto [f, g, h] = rosenboth(x);

    // element-wise probability for NaN
    constexpr double p = 1.0 / 4.0;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution{ 0, 1 };
    auto value = distribution(generator);
    if (value < p)
        f = NAN;

    g = blaze::forEach(g, [&generator, &distribution](double x) {
        auto value = distribution(generator);
        if (value < p)
            return std::nan("");
        return x;
    });

    g = blaze::forEach(g, [&generator, &distribution](double x) {
        auto value = distribution(generator);
        if (value < p)
            return std::nan("");
        return x;
    });

    h = blaze::forEach(h, [&generator, &distribution](double x) {
        auto value = distribution(generator);
        if (value < p)
            return std::nan("");
        return x;
    });

    return std::make_tuple(f, g, h);
}

cost_fun_ret_t
rosenshortg(DynamicVector<double> x)
{
    auto [f, g, h] = rosenboth(x);

    return std::make_tuple(f, DynamicVector<double>(1U, g[0]), h);
}

cost_fun_ret_t
rosenshorth(DynamicVector<double> x)
{
    auto [f, g, h] = rosenboth(x);
    return std::make_tuple(f, g, DynamicMatrix<double>(1U, 1U, h(0, 0)));
}

cost_fun_ret_t
rosennonsquarh(DynamicVector<double> x)
{
    auto [f, g, h] = rosenboth(x);
    return std::make_tuple(
      f, g, blaze::submatrix(h, 0UL, 1UL, h.rows(), h.columns() - 1));
}

using bounds_and_init_fun_t_ = std::
  tuple<DynamicVector<double>, DynamicVector<double>, DynamicVector<double>>();

using bounds_and_init_fun_t = std::function<bounds_and_init_fun_t_>;

auto
finite_bounds_include_optimum()
{
    DynamicVector<double> lb{ -2.0, -1.5 };
    DynamicVector<double> ub{ 1.5, 2.0 };
    DynamicVector<double> x0(lb.size(), 0.0F);
    return std::make_tuple(lb, ub, x0);
}

auto
finite_bounds_exlude_optimum()
{
    DynamicVector<double> lb = { -2.0, -1.5 };
    DynamicVector<double> ub = { 0.99, 0.99 };
    DynamicVector<double> x0 = (lb + ub) / 2.0;
    return std::make_tuple(lb, ub, x0);
}

auto
unbounded_and_init()
{
    DynamicVector<double> lb = { -std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity() };
    DynamicVector<double> ub = { std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity() };
    DynamicVector<double> x0 = blaze::zero<double>(lb.size());
    return std::make_tuple(lb, ub, x0);
}

class minimize1
  : public TestWithParam<
      std::tuple<std::pair<bounds_and_init_fun_t, std::string>,
                 std::pair<std::string, std::string>,
                 SubSpaceDim,
                 StepBackStrategy,
                 bool>>
{
  protected:
    void SetUp() override
    {
        options_.fatol = 0.0;
        options_.maxiter = 1000;

        std::tuple<std::string, std::string> cost_and_hess;
        std::pair<bounds_and_init_fun_t, std::string>
          bounds_and_init_fun_and_str;
        std::tie(bounds_and_init_fun_and_str,
                 cost_and_hess,
                 options_.subspace_solver,
                 options_.stepback_strategy,
                 options_.refine_stepback) = GetParam();
        bounds_and_init_fun_ = bounds_and_init_fun_and_str.first;

        auto cost_fun_str = std::get<0>(cost_and_hess);
        if (cost_fun_str == "rosengrad")
            cost_function_ = rosengrad;
        else if (cost_fun_str == "rosenboth")
            cost_function_ = rosenboth;
        else
            // invalid selection
            std::terminate();

        auto hess_approx_str = std::get<1>(cost_and_hess);
        if (hess_approx_str == "none")
            hess_approx_ = nullptr;
        else if (hess_approx_str == "sr1")
            hess_approx_ = std::make_unique<SR1>();
        else if (hess_approx_str == "bfgs")
            hess_approx_ = std::make_unique<BFGS>();
        else if (hess_approx_str == "dfp")
            hess_approx_ = std::make_unique<DFP>();
        else if (hess_approx_str == "hybrid_bfgs")
            hess_approx_ = std::make_unique<HybridUpdate>(
              std::make_unique<BFGS>(), DynamicMatrix<double>(), -1);
        else if (hess_approx_str == "hybrid_sr1")
            hess_approx_ = std::make_unique<HybridUpdate>(
              std::make_unique<SR1>(), DynamicMatrix<double>(), -1);
        else if (hess_approx_str == "hybrid_hess")
            hess_approx_ = std::make_unique<HybridUpdate>(
              std::make_unique<BFGS>(), true, -1);
        else
            // invalid selection
            std::terminate();
    }

    void TearDown() override {}

  protected:
    Options options_;
    bounds_and_init_fun_t bounds_and_init_fun_;
    cost_function_t cost_function_;
    std::unique_ptr<HessianApproximation> hess_approx_;
};

TEST_P(minimize1, test_minimize_hess_approx)
{
    auto [lb, ub, x0] = bounds_and_init_fun_();
    auto opt = Optimizer(cost_function_, lb, ub, options_, hess_approx_.get());
    auto [fval, x, grad, hess] = opt.minimize(x0);

    ASSERT_GE(opt.fval_, opt.fval_min_);

    if (opt.fval_ == opt.fval_min_) {
        assert_isclose(opt.grad_, opt.grad_min_);
        assert_isclose(opt.x_, opt.x_min_);
    }

    if (blaze::min(ub) > 1.0) {
        assert_isclose(opt.x_, std::vector<double>{ 1.0, 1.0 }, 1e-8);
        assert_isclose(
          opt.grad_, DynamicVector<double>(opt.grad_.size(), 0.0), 1e-6);
    }
}

INSTANTIATE_TEST_SUITE_P(
  test_minimize_hess_approx_parameterized,
  minimize1,
  Combine(Values(std::make_pair(finite_bounds_include_optimum,
                                "finite_bounds_include_optimum"),
                 std::make_pair(unbounded_and_init, "unbounded_and_init"),
                 std::make_pair(finite_bounds_exlude_optimum,
                                "finite_bounds_exlude_optimum")),
          Values(std::make_pair("rosenboth", "none"),
                 std::make_pair("rosengrad", "sr1"),
                 std::make_pair("rosengrad", "bfgs"),
                 std::make_pair("rosengrad", "dfp"),
                 std::make_pair("rosenboth", "hybrid_bfgs"),
                 std::make_pair("rosenboth", "hybrid_sr1"),
                 std::make_pair("rosenboth", "hybrid_hess")),
          Values(SubSpaceDim::full, SubSpaceDim::two),
          Values(StepBackStrategy::reflect,
                 StepBackStrategy::reflect_single,
                 StepBackStrategy::truncate,
                 StepBackStrategy::mixed),
          Values(false) // TODO Bool() when stepback is implemented
          ),
  [](const testing::TestParamInfo<minimize1::ParamType>& info) {
      auto bounds = std::get<0>(info.param).second;
      auto fun = std::get<1>(info.param).first;
      auto approx = std::get<1>(info.param).second;
      auto subspace_dim = subspace_dim_to_str.at(std::get<2>(info.param));
      auto stepback = step_back_strategy_str.at(std::get<3>(info.param));
      auto refine = std::to_string(std::get<4>(info.param));

      std::string name = bounds + "_" + fun + "_" + approx + "_" +
                         subspace_dim + "_" + stepback + "_" + refine;

      return name;
  });

TEST(minimize, test_multistart)
{
    auto [lb, ub, x0] = finite_bounds_exlude_optimum();
    auto fun = rosenboth;

    Options options;
    options.fatol = 0.0;
    options.refine_stepback = false;
    options.maxiter = 1000;

    std::default_random_engine generator;

    for (auto stepback :
         { StepBackStrategy::reflect, StepBackStrategy::truncate }) {
        for (auto subspace_dim : { SubSpaceDim::full, SubSpaceDim::two }) {
            options.subspace_solver = subspace_dim;
            options.stepback_strategy = stepback;

            Optimizer opt(fun, lb, ub, options, nullptr);
            for (int i = 0; i < 100; ++i) {
                auto cur_x0 =
                  blaze::map(lb, ub, [&generator](auto lb, auto ub) {
                      std::uniform_real_distribution<double> distribution(lb,
                                                                          ub);
                      return distribution(generator);
                  });
                opt.minimize(cur_x0);

                ASSERT_GE(opt.fval_, opt.fval_min_);

                if (opt.fval_ == opt.fval_min_) {
                    assert_isclose(opt.grad_, opt.grad_min_);
                    assert_isclose(opt.x_, opt.x_min_);
                }

                if (blaze::min(ub) > 1.0) {
                    assert_isclose(
                      opt.x_, std::vector<double>{ 1.0, 1.0 }, 1e-8);
                    assert_isclose(
                      opt.grad_, blaze::zero<double>(opt.grad_.size()), 1e-6);
                }
            }
        }
    }
}

TEST(minimize, test_multistart_randomfail)
{
    constexpr int n_tests = 100;

    auto [lb, ub, x0] = finite_bounds_exlude_optimum();
    Options options;
    options.fatol = 0;
    options.maxiter = 1000;
    auto opt = Optimizer(rosenrandomfail, lb, ub, options, nullptr);

    std::default_random_engine generator;

    for (int i = 0; i < n_tests; ++i) {
        auto cur_x0 = blaze::map(lb, ub, [&generator](auto lb, auto ub) {
            std::uniform_real_distribution<double> distribution(lb, ub);
            return distribution(generator);
        });
        ASSERT_THROW(opt.minimize(cur_x0), std::runtime_error);
    }
};

TEST(minimize, test_wrong_dim)
{
    auto [lb, ub, x0] = finite_bounds_exlude_optimum();
    Options options;
    options.fatol = 0;
    options.maxiter = 1000;

    std::default_random_engine generator;

    for (auto fun : { rosennonsquarh, rosenshorth, rosenshortg }) {
        auto opt = Optimizer(fun, lb, ub, options, nullptr);
        auto cur_x0 = blaze::map(lb, ub, [&generator](auto lb, auto ub) {
            std::uniform_real_distribution<double> distribution(lb, ub);
            return distribution(generator);
        });
        ASSERT_THROW(opt.minimize(cur_x0), std::runtime_error);
    }
};

TEST(minimize, test_maxiter_maxtime)
{
    auto [lb, ub, x0] = finite_bounds_exlude_optimum();
    auto fun = rosengrad;
    Options options;
    options.fatol = 0;
    DFP hessian_update;

    auto opt = Optimizer(fun, lb, ub, options, &hessian_update);

    opt.minimize(x0);
    auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now() - opt.starttime_);
    auto maxtime = time_elapsed / 10;

    opt.options_.maxiter = opt.iteration_ - 1;
    opt.minimize(x0);
    ASSERT_EQ(opt.exit_flag_, ExitStatus::max_iter);

    opt.options_.maxiter *= 10;
    opt.options_.maxtime = maxtime;
    opt.minimize(x0);
    ASSERT_EQ(opt.exit_flag_, ExitStatus::max_time);
}

TEST(debug, test_dgels)
{
    DynamicMatrix<double> A{ { -1.0, 0.0 }, { 0.0, 0.0 } };
    DynamicVector<double> b{ -2.0, 0.0 };

    DynamicVector<double> result = dgelsd(A, b);
    EXPECT_EQ(result[0], 2.0);
    EXPECT_EQ(result[1], 0.0);

    EXPECT_THROW(result = blaze::solve(A, b), std::runtime_error);
}

} // namespace
} // namespace fides
