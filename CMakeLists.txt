cmake_minimum_required(VERSION 3.31)
project(Forge)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS_DEBUG "-g1")

add_library(Forge STATIC
        "core/ops/Ops Kernels/CPU/utility_ops_CPU.cpp"
)
target_include_directories(Forge PUBLIC
        C:/Users/MSI/Documents/CPPLibs/eigen-3.4.0
        core/tensor
        C:/Users/MSI/Documents/CPPLibs/ctti/include core/
)

add_executable(main
        tests/main.cpp
)

target_link_libraries(main PRIVATE Forge)

target_precompile_headers(Forge PRIVATE
        <Eigen/Dense>
        <unsupported/Eigen/CXX11/Tensor>
)

