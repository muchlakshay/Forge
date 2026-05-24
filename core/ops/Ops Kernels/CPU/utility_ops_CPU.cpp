#include "UTILITY_OPS_CPU.h"
#include "Weights Init/weights_init_cpu.h"
#include <iostream>

void Forge::StorageBackendCPU::getStorageBackend(std::shared_ptr<StorageAbstract>& out, std::size_t size, Dtype dtype) const{
        DISPATCH_ALL_TYPES(dtype, Device::CPU, [&]{out = std::make_shared<StorageCPU<scalar_t>>(size);;});
}
void Forge::StorageBackendCPU::setView(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
    std::size_t offset, std::size_t size, Dtype dtype) const{
    DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
        dst = std::make_shared<StorageCPU<scalar_t>>(src->data(), offset, size);
    });
}


void Forge::ConstantCPU::setConstant(void* mem, std::size_t size, const Scalar& constant, Dtype dtype) const{
    DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
        auto* ptr {static_cast<scalar_t*>(mem)};
        std::visit([&](auto value) {
            using src_t = std::decay_t<decltype(value)>;
            scalar_t casted {static_cast<scalar_t>(value)};
            std::fill(ptr, ptr+size, casted);
        }, constant);
    });
}


void Forge::InitializeCPU::initialize(const void* src, void* dst, Dtype src_dtype, Dtype dst_dtype, std::size_t size) const{
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

void Forge::PrintCPU::print(const void* data_ptr, const std::vector<std::size_t>& shape, const std::vector<std::size_t>& strides,
    Dtype dtype, std::ostream& os) const {
    if (!data_ptr){std::cout<<"Tensor([])"; return;}
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


void Forge::StorageCopyCPU::copy_storage(const std::shared_ptr<StorageAbstract>& src, std::shared_ptr<StorageAbstract>& dst,
    Dtype dtype) const{
    DISPATCH_ALL_TYPES(dtype, Device::CPU, [&] {
        auto src_downcasted {std::static_pointer_cast<StorageCPU<scalar_t>>(src)};
        dst = std::make_shared<StorageCPU<scalar_t>>(src_downcasted->clone());
    });
}

void Forge::RandomCPU::randomize(Tensor &tensor, Initializers initializer) const {
    static WeightsInitCPU weights_init_cpu;
    weights_init_cpu.initialize(tensor, tensor.shape(), initializer);
}

void Forge::BroadcastAddCPU::add(const Tensor &A, const Tensor &B, const std::vector<int>& bcast_dims, Tensor &opt) const {
    DISPATCH_ALL_TYPES(A.dtype(), A.device(), [&] {
        auto A_map {A.as_eigen<scalar_t>()};
        auto B_map {B.as_eigen<scalar_t>()};
        auto opt_map {opt.as_eigen<scalar_t>()};
        Eigen::array<int64_t, 4> bcast_dims_eigen {bcast_dims[0], bcast_dims[1], bcast_dims[2], bcast_dims[3]};

        opt_map = A_map + B_map.broadcast(bcast_dims_eigen);
    });
}

void Forge::BroadcastAddGradsCPU::compute_grads(const Tensor &A, const Tensor &B, const Tensor& bcast_dims, const Tensor &opt) const {
    using grads_t = float;
    auto* data_ptr {static_cast<int*>(bcast_dims.data())};

    std::vector<Eigen::Index> rd;
    for (int i {}; i<bcast_dims.size(); ++i) {if (data_ptr[i]!=1) rd.push_back(i); }

    auto opt_grads_map {opt.gradients().as_eigen<grads_t>()};

    if (A.need_grads()) A.gradients() = opt.gradients();

    if (B.need_grads()) {
        auto B_grads_map {B.gradients().as_eigen<grads_t>()};
        switch (rd.size()) {
            case 0:
                B_grads_map = opt_grads_map;
                break;
            case 1:
                B_grads_map += opt_grads_map.sum(Eigen::array{rd[0]}).reshape(B_grads_map.dimensions());
                break;
            case 2:
                B_grads_map += opt_grads_map.sum(Eigen::array{rd[0], rd[1]}).reshape(B_grads_map.dimensions());
                break;
            case 3:
                B_grads_map += opt_grads_map.sum(Eigen::array{rd[0], rd[1], rd[2]}).reshape(B_grads_map.dimensions());
                break;
            case 4:
                B_grads_map += opt_grads_map.sum(Eigen::array{rd[0], rd[1], rd[2], rd[3]}).reshape(B_grads_map.dimensions());
                break;
            default: ;
        }
    }
}
