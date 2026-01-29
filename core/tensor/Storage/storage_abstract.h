#pragma once
#include "memory_pool_cpu.h"

class StorageAbstract {
public:
    virtual void* data() const = 0;
    [[nodiscard]] virtual std::size_t nbytes() const = 0;
    virtual ~StorageAbstract() = default;
};