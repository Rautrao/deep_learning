deep_learning_setup_eigen()

add_executable(pool src/pool.cpp)

target_link_libraries(pool PRIVATE
  deep_learning_options
  deep_learning_warnings
  deep_learning_opencv
  Eigen3::Eigen
)
