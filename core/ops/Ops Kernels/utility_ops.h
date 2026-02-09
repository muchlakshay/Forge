#pragma once
#include "enums.h"
#include "ops/Dispatcher/dispatcher.h"

namespace Forge {
    void register_utility_kernels(Dispatcher<UtilityOps>& dispatcher);
    struct StorageBackend;
    struct StorageBackendCPU;
}

struct Forge::StorageBackend : public Forge::Kernel {
    StorageBackend() : Kernel{ctti::type_id<StorageBackend>()} {}
    virtual void getStorageBackend(std::shared_ptr<StorageAbstract>&, std::size_t, Dtype) const = 0;
    ~StorageBackend() override = default;
};

struct Forge::StorageBackendCPU : public StorageBackend {
    void getStorageBackend(std::shared_ptr<StorageAbstract>& out, std::size_t size, Dtype dtype) const override{
        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&]{impl<scalar_t>(out, size);});
    }
private:
    template <TensorStorageType T>
    void impl(std::shared_ptr<StorageAbstract>& out, std::size_t size) const {
        out = std::make_shared<StorageCPU<T>>(size);
    }
};

inline void Forge::register_utility_kernels(Dispatcher<UtilityOps>& dispatcher) {
    static StorageBackendCPU storageBackendCPU{};
    dispatcher.register_kernel<StorageBackend>(&storageBackendCPU, UtilityOps::storage_backend, DispatchKey::CPU);
}


