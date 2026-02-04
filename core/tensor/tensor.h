#pragma once
#include "Storage/storage_cpu.h"
#include "../ops/Dispatcher/dispatcher.h"

namespace Forge {
    class Tensor;
    enum class Dtype {float32, float16, float64, int16, int32, int64, DtypeCount};
    enum class Device {CPU, CUDA};
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

    static Dispatcher main_dispatcher;
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

    const auto& need_grads() const {return m_need_grads;} ;
    const auto& shape() const {return m_shape;}
    const auto& strides() const {return m_strides;} ;
    const auto& dtype() const {return m_dtype;} ;
    const auto& device() const {return m_device;} ;
};