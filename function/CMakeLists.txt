deep_learning_setup_eigen()


add_executable(rotation src/rotation.cpp)
target_link_libraries(rotation PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
