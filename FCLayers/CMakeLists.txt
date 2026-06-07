deep_learning_setup_eigen()
add_executable(fcl src/main.cpp)

target_link_libraries(fcl PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
