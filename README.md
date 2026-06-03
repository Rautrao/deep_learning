# Deep Learning C++ Examples

Small C++ programs for learning deep learning building blocks.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run A Topic

Build only the target you want:

```bash
cmake --build build --target conv_basic
./build/implementing_convolutions/conv_basic
```

Other current targets:

```text
random_example
clt_example
eigen_premier
conv_basic
```

## Tests

Tests are optional, so GoogleTest is only fetched when tests are enabled:

```bash
cmake -S . -B build-tests -DBUILD_TESTING=ON
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```

## Add A New Topic

Create a new `.cpp` file in the topic folder, then add a target in that folder's
`CMakeLists.txt`.

For an Eigen-based convolution topic:

```cmake
add_executable(conv_stride src/stride.cpp)
target_link_libraries(conv_stride PRIVATE
  deep_learning_options
  deep_learning_warnings
  Eigen3::Eigen
)
```

Shared dependencies live in `cmake/Dependencies.cmake`, so libraries are set up
once and reused by the targets that need them.
