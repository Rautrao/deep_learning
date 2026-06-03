deep_learning_setup_opencv()
deep_learning_setup_eigen()
add_executable(opencv_basic src/main.cpp)

target_link_libraries(opencv_basic PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
