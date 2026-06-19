deep_learning_setup_eigen()

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
