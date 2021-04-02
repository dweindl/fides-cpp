cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074)
    # Use package_ROOT environment variables
    cmake_policy(SET CMP0074 NEW)
endif(POLICY CMP0074)

project(fides LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------------------------------------------------------
# GSL lite
# ------------------------------------------------------------------------------
set(GSL_LITE_URL https://github.com/gsl-lite/gsl-lite.git)

include(ExternalProject)
find_package(Git REQUIRED)

set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/3rd_party)

ExternalProject_Add(
    gsl-extern
    GIT_REPOSITORY ${GSL_LITE_URL}
    TIMEOUT 10
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON)
ExternalProject_Get_Property(gsl-extern SOURCE_DIR)
set(GSL_LITE_INCLUDE_DIR
    ${SOURCE_DIR}/include
    CACHE INTERNAL "Include folder for gsl-lite")

add_library(gsl INTERFACE)
target_include_directories(gsl INTERFACE ${GSL_LITE_INCLUDE_DIR})

add_library(gsl::gsl-lite ALIAS gsl)

install(TARGETS gsl EXPORT FidesTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# ------------------------------------------------------------------------------
# spdlog
# ------------------------------------------------------------------------------

find_package(spdlog REQUIRED)

# ------------------------------------------------------------------------------
# Blaze
# ------------------------------------------------------------------------------

find_package(blaze REQUIRED)
add_library(blaze_target INTERFACE)
target_link_libraries(blaze_target INTERFACE blaze::blaze)

# ------------------------------------------------------------------------------
# LAPACK
# ------------------------------------------------------------------------------

find_package(LAPACK REQUIRED)

# ------------------------------------------------------------------------------
# ...
# ------------------------------------------------------------------------------
include(version)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

# ------------------------------------------------------------------------------
# Export
# ------------------------------------------------------------------------------
install(
    TARGETS ${PROJECT_NAME}
    EXPORT FidesTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fides)

export(
    EXPORT FidesTargets
    FILE FidesTargets.cmake
    NAMESPACE Fides::)
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/FidesConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/FidesConfig.cmake"
    INSTALL_DESTINATION "${LIB_INSTALL_DIR}/cmake/")
# TODO
set(PROJECT_VERSION 0.0.0)
write_basic_package_version_file(FidesConfigVersion.cmake
                                 COMPATIBILITY ExactVersion)
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/FidesConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/FidesTargets.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/FidesConfigVersion.cmake
    DESTINATION share/Fides/cmake)

option(EXPORT_PACKAGE "Export Fides library to CMake package registry?" ON)
if(EXPORT_PACKAGE)
    export(PACKAGE Fides)
endif()
