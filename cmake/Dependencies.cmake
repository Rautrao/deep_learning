include(FetchContent)

function(deep_learning_setup_eigen)
  if(TARGET Eigen3::Eigen)
    return()
  endif()

  find_package(Eigen3 QUIET)

  if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
      Eigen3
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 6e4d5d4832a1736ba2f6faa7087b61227dba8e7b
    )
    FetchContent_MakeAvailable(Eigen3)
  endif()

  if(NOT TARGET Eigen3::Eigen AND TARGET eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
  endif()
endfunction()

function(deep_learning_setup_googletest)
  if(TARGET GTest::gtest_main)
    return()
  endif()

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endfunction()

function(deep_learning_setup_opencv)
  if(TARGET opencv_core)
    return()
  endif()

  find_package(OpenCV REQUIRED)

  add_library(deep_learning_opencv INTERFACE)
  target_include_directories(deep_learning_opencv INTERFACE
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(deep_learning_opencv INTERFACE
    ${OpenCV_LIBS}
  )
endfunction()
