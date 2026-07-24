deep_learning_setup_opencv()
deep_learning_setup_eigen()
add_executable(gda src/main.cpp)

target_link_libraries(gda PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
