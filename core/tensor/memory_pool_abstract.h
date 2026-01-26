#pragma once

class MemoryPoolAbstract {
    public:
    using byte = unsigned char;
    virtual void* getMem(std::size_t) = 0;
    virtual void returnMem(void*) = 0;
    virtual ~MemoryPoolAbstract() = default;
};