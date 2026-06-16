deep_learning_setup_eigen()


add_executable(bce src/bce.cpp)
target_link_libraries(bce PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
