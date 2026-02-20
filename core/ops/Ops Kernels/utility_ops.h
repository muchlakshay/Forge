#pragma once
#include "enums.h"
#include "ops/Dispatcher/dispatcher.h"

namespace Forge {
    void register_utility_kernels(Dispatcher<UtilityOps>& dispatcher);
    struct StorageBackend;
    struct StorageBackendCPU;
    struct ConstantAbstract;
    struct ConstantCPU;
    struct InitializeAbstract;
    struct InitializeCPU;
    struct PrintAbstract;
    struct PrintCPU;
    struct StorageCopyAbstract;
    struct StorageCopyCPU;

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
    virtual void setView(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        std::size_t offset, std::size_t size, Dtype dtype) const = 0;
    ~StorageBackend() override = default;
};

struct Forge::StorageBackendCPU : public StorageBackend {
    void getStorageBackend(std::shared_ptr<StorageAbstract>& out, std::size_t size, Dtype dtype) const override{
        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&]{out = std::make_shared<StorageCPU<scalar_t>>(size);;});
    }
    void setView(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        std::size_t offset, std::size_t size, Dtype dtype) const override {
        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
            dst = std::make_shared<StorageCPU<scalar_t>>(src->data(), offset, size);
        });
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

struct Forge::InitializeAbstract : public Forge::Kernel {
    InitializeAbstract() : Kernel{ctti::type_id<InitializeAbstract>()} {}
    virtual void initialize(const void* src, void* dst, Dtype src_dtype, Dtype dst_dtype, std::size_t size) const = 0;
};

struct Forge::InitializeCPU : public InitializeAbstract {
    void initialize(const void* src, void* dst, Dtype src_dtype, Dtype dst_dtype, std::size_t size) const override {
        DISPATCH_ALL_TYPES(dst_dtype, Device::CPU, [&] {
            using dst_t = scalar_t;
            auto* dst_ptr {static_cast<dst_t*>(dst)};
            DISPATCH_ALL_TYPES(src_dtype, Device::CPU, [&] {
                using src_t = scalar_t;
                auto* src_ptr {static_cast<const src_t*>(src)};
                for (std::size_t i {}; i<size; ++i) dst_ptr[i] = static_cast<dst_t>(src_ptr[i]);
            });
        });
    }
};

struct Forge::PrintAbstract : public Forge::Kernel {
    PrintAbstract() : Kernel{ctti::type_id<PrintAbstract>()} {}
    virtual void print(const void* data_ptr, const std::vector<std::size_t>& shape,const std::vector<std::size_t>& strides,
        Dtype dtype, std::ostream& os) const = 0;
};

struct Forge::PrintCPU : public PrintAbstract {
    void print(const void* data_ptr, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides,
        Dtype dtype, std::ostream& os) const override {
        if (shape.empty()) {
            DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {os << "Tensor(["<<*static_cast<const scalar_t*>(data_ptr)<<"])";});
            return;
        }
        os << "Tensor(shape=";
        for (size_t i = 0; i < shape.size(); ++i) { os << shape[i]; if (i + 1 != shape.size()) os << "x";}
        os << "):\n";

        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
            print_tensor_impl<scalar_t>(os, data_ptr, shape, strides);
        });
    }
    template<typename T>
    void print_tensor_impl(std::ostream& os, const void* data_ptr, const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& strides) const {
        const T* data = static_cast<const T*>(data_ptr);
        print_recursive(os,data,shape,strides,0,0);
    }
    template<typename T>
    void print_recursive(std::ostream& os, const T* data, const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& strides, std::size_t dim, std::size_t offset
    ) const {
            os << "[";
            if (dim == shape.size() - 1) {
                for (std::size_t i = 0; i < shape[dim]; ++i) {
                    os << data[offset + i * strides[dim]];
                    if (i + 1 != shape[dim]) os << ", ";
                }
            }
            else {
                for (std::size_t i = 0; i < shape[dim]; ++i) {
                    print_recursive(os,data,shape,strides,dim + 1, offset + i * strides[dim]);
                    if (i + 1 != shape[dim]) { os << ",\n" << std::string(dim + 1, ' ');}
                }
            }
            os << "]";
        }
};

struct Forge::StorageCopyAbstract : public Kernel {
    StorageCopyAbstract() : Kernel{ctti::type_id<StorageCopyAbstract>()} {}
    virtual void copy_storage(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        Dtype dtype) const = 0;
};

struct Forge::StorageCopyCPU final : public StorageCopyAbstract {
    void copy_storage(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        Dtype dtype) const override {
        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
            auto src_downcasted {std::static_pointer_cast<StorageCPU<scalar_t>>(src)};
            dst = std::make_shared<StorageCPU<scalar_t>>(src_downcasted->clone());
        });
    }
};

inline void Forge::register_utility_kernels(Dispatcher<UtilityOps>& dispatcher) {
    static StorageBackendCPU storageBackendCPU{};
    static ConstantCPU constantCPU{};
    static InitializeCPU initializeCPU{};
    static PrintCPU printCPU{};
    dispatcher.register_kernel<StorageBackend>(&storageBackendCPU, UtilityOps::storage_backend, DispatchKey::CPU);
    dispatcher.register_kernel<ConstantAbstract>(&constantCPU, UtilityOps::constant, DispatchKey::CPU);
    dispatcher.register_kernel<InitializeAbstract>(&initializeCPU, UtilityOps::initializers, DispatchKey::CPU);
    dispatcher.register_kernel<PrintAbstract>(&printCPU, UtilityOps::print, DispatchKey::CPU);
}