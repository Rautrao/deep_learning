add_library(deep_learning_options INTERFACE)
target_compile_features(deep_learning_options INTERFACE cxx_std_17)

add_library(deep_learning_warnings INTERFACE)
target_compile_options(deep_learning_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -pedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
