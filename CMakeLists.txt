cmake_minimum_required(VERSION 3.18)
project(medimg_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(medimg_cuda
    src/main.cu
)

target_include_directories(medimg_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)

target_compile_options(medimg_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(medimg_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
