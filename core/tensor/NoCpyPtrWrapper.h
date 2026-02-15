#pragma once
#include <memory>

namespace Forge {
    template<typename T>
    struct SkipCopyPtr;
}

template<typename T>
struct Forge::SkipCopyPtr {
    std::unique_ptr<T> m_ptr {};
    explicit SkipCopyPtr(std::unique_ptr<T> ptr=nullptr) : m_ptr{std::move(ptr)} {}
    SkipCopyPtr(const SkipCopyPtr&) {}
    SkipCopyPtr& operator=(const SkipCopyPtr&) { return *this; }
    T* operator->() { return m_ptr.get(); }
    T& operator*() { return *m_ptr; }
    explicit operator bool() const { return m_ptr != nullptr; }
};