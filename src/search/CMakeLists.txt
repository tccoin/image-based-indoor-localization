cmake_minimum_required(VERSION 3.16)
project(search CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_library(search SHARED lib.cu)

add_executable(test main.cu)
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test search)
