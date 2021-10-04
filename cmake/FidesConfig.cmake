@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

include("${CMAKE_CURRENT_LIST_DIR}/FidesTargets.cmake")

find_dependency(spdlog REQUIRED HINTS "@spdlog_DIR@")
find_dependency(blaze REQUIRED HINTS "@blaze_DIR@")
find_dependency(LAPACK REQUIRED)

check_required_components(Fides)
