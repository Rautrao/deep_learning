add_executable(random_example random_example.cpp)
target_link_libraries(random_example PRIVATE
  deep_learning_options
  deep_learning_warnings
)

add_executable(clt_example clt_example.cpp)
target_link_libraries(clt_example PRIVATE
  deep_learning_options
  deep_learning_warnings
)

deep_learning_setup_eigen()

add_executable(eigen_premier eigen_premier.cpp)
target_link_libraries(eigen_premier PRIVATE
  deep_learning_options
  deep_learning_warnings
  Eigen3::Eigen
)
