#pragma once
#include "Storage/storage_cpu.h"
#include "enums.h"
#include "../ops/Dispatcher/dispatcher.h"
#include "../ops/Ops Kernels/kernels_registrar.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <ranges>
#include <memory>

namespace Forge {
    class Tensor;
}

template <typename T>
using init_list_1D = std::initializer_list<T>;
template <typename T>
using init_list_2D = std::initializer_list<init_list_1D<T>>;
template <typename T>
using init_list_3D = std::initializer_list<init_list_2D<T>>;
template <typename T>
using init_list_4D = std::initializer_list<init_list_3D<T>>;

class Forge::Tensor {
    Device m_device{};
    Dtype m_dtype{};
    std::shared_ptr<StorageAbstract> m_storage{};
    std::vector<std::size_t> m_shape{}, m_strides{};
    std::size_t m_size{};
    bool m_need_grads{};
    DispatchKey m_dispatch_key{};

    static Dispatcher<UtilityOps>& dispatcher();
public:
    explicit Tensor(const std::vector<std::size_t>& shape, Forge::Dtype dtype = Forge::Dtype::float32, bool need_grads = true,
        Forge::Device device = Forge::Device::CPU);

    template <typename T>
    void setValues(init_list_1D<T>);
    template <typename T>
    void setValues(init_list_2D<T>);
    template <typename T>
    void setValues(init_list_3D<T>);
    template <typename T>
    void setValues(init_list_4D<T>);

    template <typename T>
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> as_eigen() const;

    [[nodiscard]] const auto& need_grads() const {return m_need_grads;} ;
    [[nodiscard]] const auto& shape() const {return m_shape;}
    [[nodiscard]] const auto& strides() const {return m_strides;} ;
    [[nodiscard]] const auto& dtype() const {return m_dtype;} ;
    [[nodiscard]] const auto& device() const {return m_device;} ;
    [[nodiscard]] void* data() const {return m_storage->data();}
};



inline Forge::Dispatcher<Forge::UtilityOps>& Forge::Tensor::dispatcher() {
    static Dispatcher<UtilityOps> utility_dispatcher;
    static bool is_defined = false;
    if (is_defined) return utility_dispatcher;
    Forge::register_all_kernels(utility_dispatcher);
    is_defined = true;
    return utility_dispatcher;
}

inline Forge::Tensor::Tensor(const std::vector<std::size_t>& shape, Forge::Dtype dtype, bool need_grads,
        Forge::Device device) : m_device{device}, m_dtype{dtype}, m_shape{shape}, m_strides(shape.size(), 0), m_need_grads{need_grads},
        m_dispatch_key {device==Device::CPU?DispatchKey::CPU : DispatchKey::CUDA}{

    std::size_t size {1u};
    for (int i = shape.size()-1; i>=0; i--) {
        m_strides[i] = size;
        size*=m_shape[i];
    }
    m_size = size;
    const auto storage_backend { dispatcher().lookup<StorageBackend>(UtilityOps::storage_backend, m_dispatch_key)};
    storage_backend->getStorageBackend(m_storage, size, m_dtype);
}

