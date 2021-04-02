# The main fides library

set(SRC_LIST
    fides.cpp
    hessian_approximation.cpp
    minimize.cpp
    stepback.cpp
    steps.cpp
    subproblem.cpp
    trust_region.cpp)

add_library(${PROJECT_NAME} ${SRC_LIST})

target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    #PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    )

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC blaze::blaze
    PUBLIC gsl::gsl-lite
    PRIVATE spdlog::spdlog
    PUBLIC ${LAPACK_LIBRARIES})
# CMake 3.18: PUBLIC LAPACK::LAPACK

file(GLOB PUBLIC_HEADERS ../include/fides/*.hpp)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")
