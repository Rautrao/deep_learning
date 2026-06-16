deep_learning_setup_eigen()


add_executable(gradient src/gradient.cpp)
target_link_libraries(gradient PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
