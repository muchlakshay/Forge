#pragma once
#include "enums.h"
#include "ops/Dispatcher/dispatcher.h"

namespace Forge {
    void register_utility_kernels(Dispatcher<UtilityOps>& dispatcher);
    struct StorageBackend;
    struct StorageBackendCPU;
    struct ConstantAbstract;
    struct ConstantCPU;

    using Scalar = std::variant<
    float,
    double,
    std::int16_t,
    std::int32_t,
    std::int64_t
>;
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

struct Forge::ConstantAbstract : public Forge::Kernel {
  ConstantAbstract() : Kernel{ctti::type_id<ConstantAbstract>()} {}
  virtual void setConstant(void* mem, std::size_t size, const Scalar& constant, Dtype dtype) const = 0;
    ~ConstantAbstract() override = default;
};

struct Forge::ConstantCPU : public ConstantAbstract {
    void setConstant(void* mem, std::size_t size, const Scalar& constant, Dtype dtype) const override {
        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
            auto* ptr {static_cast<scalar_t*>(mem)};
            std::visit([&](auto value) {
                using src_t = std::decay_t<decltype(value)>;
                scalar_t casted {static_cast<scalar_t>(value)};
                std::fill(ptr, ptr+size, casted);
            }, constant);
        });
    }
};

inline void Forge::register_utility_kernels(Dispatcher<UtilityOps>& dispatcher) {
    static StorageBackendCPU storageBackendCPU{};
    static ConstantCPU constantCPU{};
    dispatcher.register_kernel<StorageBackend>(&storageBackendCPU, UtilityOps::storage_backend, DispatchKey::CPU);
    dispatcher.register_kernel<ConstantAbstract>(&constantCPU, UtilityOps::constant, DispatchKey::CPU);
}


