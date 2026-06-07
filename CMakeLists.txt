cmake_minimum_required(VERSION 3.20)

project(deep_learning LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build test targets" OFF)

if(BUILD_TESTING)
  enable_testing()
endif()

include(cmake/ProjectOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(basic_programs)
add_subdirectory(implementing_convolutions)
add_subdirectory(googletest_example)
add_subdirectory(using_opencv)
add_subdirectory(using_tensors)
add_subdirectory(FCLayers)
