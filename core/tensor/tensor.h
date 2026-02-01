#pragma once
#include "Storage/storage_cpu.h"

namespace Forge {
    class Tensor{};
    enum class Dtype {float32, float16, float64, int16, int32, int64};
    enum Device {CPU, CUDA};
}

class Forge::Tensor {
    Device m_device{};
    Dtype m_dtype{};
    std::shared_ptr<StorageAbstract> m_storage{};
    std::vector<std::size_t> m_shape{}, m_strides{};
    std::size_t m_size{};
    bool m_need_grads{};
};