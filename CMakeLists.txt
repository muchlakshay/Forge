cmake_minimum_required(VERSION 3.31)
project(Forge)

set(CMAKE_CXX_STANDARD 20)

add_library(Forge STATIC
        "core/ops/Ops Kernels/CPU/utility_ops_CPU.cpp"
        core/tensor/Storage/CPU/memory_pool_cpu.cpp
        core/tensor/tensor.cpp
        primitives/Activations/activations.cpp
        primitives/Activations/CPU/activationsCPU.cpp
        primitives/Activations/CPU/activationsGradsCPU.cpp
        primitives/Linear/CPU/LinearGradsCPU.cpp
        primitives/Linear/LinearLayer.cpp
        primitives/Linear/CPU/LinearLayerCPU.cpp
        "primitives/Weights Init/weights_init_cpu.cpp"
        "primitives/Loss Functions/lossFunctions.cpp"
        "primitives/Loss Functions/CPU/lossFunctionsCPU.cpp"
        "primitives/Loss Functions/CPU/lossFunctionsGrads.cpp"
        primitives/Optimizers/optimizers.cpp
        primitives/Optimizers/CPU/optimizersUpdateCPU.cpp
)

target_include_directories(Forge PUBLIC
        C:/Users/MSI/Documents/CPPLibs/eigen-3.4.0
        core/tensor
        C:/Users/MSI/Documents/CPPLibs/ctti/include core/
        core/
        primitives/
)

add_executable(main
        tests/main.cpp
)

target_link_libraries(main PRIVATE Forge)

target_precompile_headers(Forge PRIVATE
        <Eigen/Dense>
        <unsupported/Eigen/CXX11/Tensor>
)

