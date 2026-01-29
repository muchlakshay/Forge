#pragma once

namespace Forge {
    class MemoryPoolAbstract;
    class MemoryPoolCPU;
}

class Forge::MemoryPoolAbstract {
    public:
    using byte = unsigned char;
    virtual void* allocate(std::size_t) = 0;
    virtual void free(void*) = 0;
    virtual ~MemoryPoolAbstract() = default;
};