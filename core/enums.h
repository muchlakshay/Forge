#pragma once

namespace Forge {
    enum class Dtype {float32, float16, float64, int16, int32, int64, DtypeCount};
    enum class Device {CPU, CUDA};
    enum class DispatchKey {CPU, CUDA, CPU_Autodiff, CUDA_Autodiff, DispatchKeysCount};
    enum class UtilityOps {storage_backend, initializers, constant, print, storage_copy, Count};
}