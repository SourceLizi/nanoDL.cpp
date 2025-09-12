cmake_minimum_required(VERSION 3.10.0)
project(nanoDL VERSION 0.1.0 LANGUAGES C CXX)

include_directories(src/)
set(CMAKE_C_FLAGS_DEBUG "-g")

# add_executable(nanoDL)
# file(GLOB sources main.cpp src/*.cpp src/*.h)
# target_sources(nanoDL PUBLIC ${sources})

include(CTest)
enable_testing()

add_executable(tests tests/test_main.cpp)
file(GLOB tests_sources tests/*.cpp tests/*.h src/*.cpp)
target_sources(tests PUBLIC ${tests_sources})
add_test(tensor_tests tests 0)
add_test(autograd_scalar_test tests 1)
add_test(autograd_tensor_test tests 2)