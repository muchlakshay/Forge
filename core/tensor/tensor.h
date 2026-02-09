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
