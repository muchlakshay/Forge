#pragma once

namespace Forge {
    // enum class Dtype {float32, float16, float64, int16, int32, int64, DtypeCount};
    enum class Dtype {float32, int32, DtypeCount};
    enum class Device {CPU, CUDA};
    enum class DispatchKey {CPU, CUDA, CPU_Autodiff, CUDA_Autodiff, DispatchKeysCount};
    enum class UtilityOps {storage_backend, initializers, constant, print, storage_copy, random, bcast_add, Count};
    enum class Initializers{xavier_normal, xavier_uniform, he_normal, he_uniform};
}