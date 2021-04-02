#ifndef FIDES_TESTS_MISC_HPP
#define FIDES_TESTS_MISC_HPP

#include "gtest/gtest.h"

template< typename T>
constexpr decltype(auto) assert_almost_equal( T const& v1, T const& v2)
{
    ASSERT_EQ(v1.size(), v2.size());
    for(std::size_t i=0UL; i<v1.size(); ++i ) {
        ASSERT_DOUBLE_EQ(v1[i], v2[i]);
    }
}

template< typename T1, typename T2>
constexpr decltype(auto) assert_near( T1 const& v1, T2 const& v2, double atol = 1e-8)
{
    ASSERT_EQ(v1.size(), v2.size());
    for(std::size_t i=0UL; i<v1.size(); ++i ) {
        ASSERT_NEAR(v1[i], v2[i], atol);
    }
}

template< typename T1, typename T2>
constexpr decltype(auto) assert_isclose( T1 const& v1, T2 const& v2,
                                       double atol = 1e-8, double rtol=1e-5 )
{
    ASSERT_EQ(v1.size(), v2.size());
    for(std::size_t i=0UL; i<v1.size(); ++i ) {
        ASSERT_LE(std::abs(v1[i] - v2[i]), atol + rtol * std::abs(v1[i]));
    }
}

#endif // FIDES_TESTS_MISC_HPP
