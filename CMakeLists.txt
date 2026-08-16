cmake_minimum_required(VERSION 3.20)
project(Forge VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(FORGE_TESTS_BUILD ON CACHE BOOL "Enable Forge's test builds")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen GIT_TAG 3.4.0)
FetchContent_Declare(reflectcpp GIT_REPOSITORY https://github.com/getml/reflect-cpp GIT_TAG v0.25.0)
FetchContent_Declare(ctti GIT_REPOSITORY https://github.com/Manu343726/ctti)

FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(reflectcpp)
FetchContent_MakeAvailable(ctti)

find_package(OpenMP REQUIRED)

if(WIN32)
    set(_openblas_root_default "C:/Users/MSI/Documents/CPPLibs/OpenBLAS")
else()
    set(_openblas_root_default "/usr")
endif()
set(OPENBLAS_ROOT "${_openblas_root_default}" CACHE PATH "Path to OpenBLAS install")

find_path(OPENBLAS_INCLUDE_DIR cblas.h PATHS "${OPENBLAS_ROOT}/include")
find_library(OPENBLAS_LIBRARY
        NAMES openblas libopenblas.dll libopenblas
        PATHS "${OPENBLAS_ROOT}/lib"
)

if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
    message(FATAL_ERROR "OpenBLAS not found under OPENBLAS_ROOT='${OPENBLAS_ROOT}'. "
            "Pass -DOPENBLAS_ROOT=<path> pointing at the folder containing include/ and lib/.")
endif()

add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
        IMPORTED_LOCATION "${OPENBLAS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
)

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
        "primitives/Attention/Self Attention/self_attention.cpp"
        "primitives/Attention/Self Attention/CPU/self_attention_cpu.cpp"
        "primitives/Attention/Self Attention/CPU/self_attention_grads_cpu.cpp"
        primitives/LayerNorm/layernorm.cpp
        primitives/LayerNorm/CPU/layernorm_cpu.cpp
        primitives/LayerNorm/CPU/layernorm_grads_cpu.cpp
        "primitives/Sinusoidal PE/sinusoidal_pe.cpp"
        primitives/Embeddings/CPU/embeddings_cpu.cpp
        primitives/Embeddings/embeddings.cpp
        primitives/Embeddings/CPU/embeddings_grads_cpu.cpp
        data/preprocessing/tokenizer.cpp
)

target_link_libraries(Forge PUBLIC Eigen3::Eigen OpenBLAS::OpenBLAS reflectcpp::reflectcpp OpenMP::OpenMP_CXX)

target_compile_definitions(Forge PRIVATE EIGEN_USE_THREADS EIGEN_USE_OPENMP)

target_include_directories(Forge
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives
        ${ctti_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(Forge PUBLIC /arch:AVX2 /fp:fast /openmp)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Forge PUBLIC -mavx2 -march=native -ffast-math -fopenmp)
    target_link_options(Forge PUBLIC $<$<CXX_COMPILER_ID:GNU,Clang>:-fopenmp>)
endif()

if(MINGW)
    target_compile_options(Forge PUBLIC -Wa,-mbig-obj)
endif()

target_precompile_headers(Forge PUBLIC
        <Eigen/Dense>
        <unsupported/Eigen/CXX11/Tensor>
)

if (${FORGE_TESTS_BUILD})
    add_subdirectory(experiments)
endif()