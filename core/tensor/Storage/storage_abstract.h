#pragma once
#include "CPU/memory_pool_cpu.h"

template <typename T>
concept TensorStorageType = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

namespace Forge {
    class StorageAbstract;
}

class Forge::StorageAbstract {
public:
    [[nodiscard]] virtual void* data() const = 0;
    [[nodiscard]] virtual std::size_t nbytes() const = 0;
    virtual ~StorageAbstract() = default;
};