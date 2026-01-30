#pragma once
#include "memory_pool_cpu.h"

namespace Forge {
    class StorageAbstract;
}

class Forge::StorageAbstract {
public:
    virtual void* data() const = 0;
    [[nodiscard]] virtual std::size_t nbytes() const = 0;
    virtual ~StorageAbstract() = default;
};