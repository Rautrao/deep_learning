add_library(googletest_example_lib src/my_functions.cpp)
target_include_directories(googletest_example_lib PUBLIC include)
target_link_libraries(googletest_example_lib PUBLIC deep_learning_options)
target_link_libraries(googletest_example_lib PRIVATE deep_learning_warnings)

if(BUILD_TESTING)
  deep_learning_setup_googletest()

  add_executable(test_googletest_example tests/test_my_function.cpp)
  target_link_libraries(test_googletest_example PRIVATE
    deep_learning_options
    deep_learning_warnings
    googletest_example_lib
    GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(test_googletest_example)
endif()
