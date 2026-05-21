#pragma once
#include "enums.h"
#include "ops/Dispatcher/dispatcher.h"
#include "Storage/storage_abstract.h"

namespace Forge {
    void register_utility_kernels(Dispatcher<UtilityOps>& dispatcher);
    struct StorageBackend;
    struct ConstantAbstract;
    struct InitializeAbstract;
    struct PrintAbstract;
    struct StorageCopyAbstract;
    struct RandomAbstract;

    class Tensor;

    using Scalar = std::variant<
    float,
    double,
    std::int16_t,
    std::int32_t,
    std::int64_t
>;
}

struct Forge::StorageBackend : Kernel {
    StorageBackend() : Kernel{ctti::type_id<StorageBackend>()} {}
    virtual void getStorageBackend(std::shared_ptr<StorageAbstract>&, std::size_t, Dtype) const = 0;
    virtual void setView(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        std::size_t offset, std::size_t size, Dtype dtype) const = 0;
    ~StorageBackend() override = default;
};


struct Forge::ConstantAbstract : Kernel {
  ConstantAbstract() : Kernel{ctti::type_id<ConstantAbstract>()} {}
  virtual void setConstant(void* mem, std::size_t size, const Scalar& constant, Dtype dtype) const = 0;
    ~ConstantAbstract() override = default;
};


struct Forge::InitializeAbstract : Kernel {
    InitializeAbstract() : Kernel{ctti::type_id<InitializeAbstract>()} {}
    virtual void initialize(const void* src, void* dst, Dtype src_dtype, Dtype dst_dtype, std::size_t size) const = 0;
};

struct Forge::PrintAbstract : Kernel {
    PrintAbstract() : Kernel{ctti::type_id<PrintAbstract>()} {}
    virtual void print(const void* data_ptr, const std::vector<std::size_t>& shape,const std::vector<std::size_t>& strides,
        Dtype dtype, std::ostream& os) const = 0;
};

struct Forge::StorageCopyAbstract : Kernel {
    StorageCopyAbstract() : Kernel{ctti::type_id<StorageCopyAbstract>()} {}
    virtual void copy_storage(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        Dtype dtype) const = 0;
};

struct Forge::RandomAbstract : Kernel {
    RandomAbstract() : Kernel{ctti::type_id<RandomAbstract>()} {}
    virtual void randomize(Tensor& tensor, Initializers initializer) const = 0;
};