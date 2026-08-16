#pragma once
#include "../utility_ops.h"
#include "../../../tensor/Storage/CPU/storage_cpu.h"
#include <memory>

namespace Forge {
    struct StorageBackendCPU;
    struct ConstantCPU;
    struct InitializeCPU;
    struct PrintCPU;
    struct StorageCopyCPU;
    struct RandomCPU;
    struct BroadcastAddCPU;
    struct BroadcastAddGradsCPU;
};

struct Forge::StorageBackendCPU : StorageBackend {
    void getStorageBackend(std::shared_ptr<StorageAbstract>& out, std::size_t size, Dtype dtype) const override;
    void setView(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        std::size_t offset, std::size_t size, Dtype dtype) const override;
};

struct Forge::ConstantCPU : ConstantAbstract {
    void setConstant(void* mem, std::size_t size, const Scalar& constant, Dtype dtype) const override;
};

struct Forge::InitializeCPU : InitializeAbstract {
    void initialize(const void* src, void* dst, Dtype src_dtype, Dtype dst_dtype, std::size_t size) const override;
};

struct Forge::PrintCPU : PrintAbstract {
    void print(const void* data_ptr, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides,
        Dtype dtype, std::ostream& os) const override;
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

struct Forge::StorageCopyCPU final : StorageCopyAbstract {
    void copy_storage(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
        Dtype dtype) const override;
};

struct Forge::RandomCPU final : RandomAbstract {
    void randomize(Tensor& tensor, Initializers initializer) const override;
};

struct Forge::BroadcastAddCPU : BroadcastAddAbstract {
  void add(const Tensor &A, const Tensor &B, const std::vector<int>& bcast_dims,  Tensor &opt) const override;
};

struct Forge::BroadcastAddGradsCPU : BroadcastAddGradsAbstract {
    void compute_grads(const Tensor &A, const Tensor &B, const Tensor& bcast_dims, const Tensor &opt) const override;
};