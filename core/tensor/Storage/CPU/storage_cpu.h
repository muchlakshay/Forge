#pragma once
#include <cstring>
#include "../storage_abstract.h"
#include "memory_pool_cpu.h"

namespace Forge {
    template<TensorStorageType T>
    class StorageCPU;

    inline MemoryPoolCPU& cpu_memory_pool() {
        static MemoryPoolCPU pool;
        return pool;
    }
}

template <TensorStorageType T>
class Forge::StorageCPU final : public StorageAbstract {
    bool m_is_view {};
    T* m_data_ptr{};
    std::size_t m_size {};
public:
    explicit StorageCPU(std::size_t size);
    explicit StorageCPU(void* another, std::size_t offset, std::size_t size);

    StorageCPU(StorageCPU&& another) noexcept;
    StorageCPU& operator=(StorageCPU&& another) noexcept;

    StorageCPU(const StorageCPU&) = delete;
    StorageCPU& operator=(const StorageCPU&) = delete;

    ~StorageCPU() override {if (!m_is_view && m_data_ptr) cpu_memory_pool().free(m_data_ptr);};
    StorageCPU clone() const;

    [[nodiscard]] void* data() const override {return m_data_ptr;}
    T* data_() const {return m_data_ptr;}
    [[nodiscard]] std::size_t nbytes() const override {return m_size*sizeof(T);}
    [[nodiscard]] std::size_t size() const {return m_size;}
};

template <TensorStorageType T>
Forge::StorageCPU<T>::StorageCPU(const std::size_t size)
        : m_data_ptr(static_cast<T*>(cpu_memory_pool().allocate(size*sizeof(T)))), m_size(size) {}
template<TensorStorageType T>
Forge::StorageCPU<T>::StorageCPU(void *another, const std::size_t offset, const std::size_t size)
    : m_is_view{true}, m_data_ptr{static_cast<T*>(another)+offset}, m_size{size} {}
template<TensorStorageType T>
Forge::StorageCPU<T>::StorageCPU(StorageCPU&& another) noexcept
    : m_is_view{another.m_is_view}, m_data_ptr{another.m_data_ptr}, m_size{another.m_size}{
    another.m_data_ptr = nullptr;
    another.m_size = 0;
}
template<TensorStorageType T>
Forge::StorageCPU<T>& Forge::StorageCPU<T>::operator=(StorageCPU&& another) noexcept {
    if (this == &another) return *this;
    if (!m_is_view && m_data_ptr) cpu_memory_pool().free(m_data_ptr);
    m_is_view = another.m_is_view;
    m_data_ptr = std::exchange(another.m_data_ptr, nullptr);
    m_size = std::exchange(another.m_size, 0);
    return *this;
}

template<TensorStorageType T>
Forge::StorageCPU<T> Forge::StorageCPU<T>::clone() const {
    StorageCPU clone{m_size};
    std::memcpy(clone.m_data_ptr, m_data_ptr, m_size*sizeof(T));
    return clone;
}