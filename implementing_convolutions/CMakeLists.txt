deep_learning_setup_eigen()

add_executable(conv_basic src/main.cpp)
target_link_libraries(conv_basic PRIVATE
  deep_learning_options
  deep_learning_warnings
  Eigen3::Eigen
)
