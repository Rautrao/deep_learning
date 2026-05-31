add_test([=[SigmoidTest.BaseCase]=]  /home/ambadas/projects/deep_learning/googletest_example/build/test_googletest_example [==[--gtest_filter=SigmoidTest.BaseCase]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[SigmoidTest.BaseCase]=]  PROPERTIES WORKING_DIRECTORY /home/ambadas/projects/deep_learning/googletest_example/build SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_googletest_example_TESTS SigmoidTest.BaseCase)
