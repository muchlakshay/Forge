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

    template<typename T>
    void infer_shape(const T&, const std::vector<std::size_t>&);
    template<typename T>
    void infer_shape(const std::initializer_list<T>& init_list, std::vector<std::size_t>& shape);

    template <typename T, typename U>
    void flatten(const T& item, std::vector<U>& result);
    template<typename T, typename U>
    void flatten(const std::initializer_list<T>& list, std::vector<U>& result);
    template<typename T, typename U>
    std::vector<T> flatten_list(const std::initializer_list<U>& list);

    template <TensorStorageType T>
    constexpr Dtype dtype_of();
    template<typename T>
    inline constexpr bool dependent_false_v {false};

    template<typename T>
    void check_shape_and_throw(const std::initializer_list<T>& init_list, const std::vector<std::size_t>& shape);
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

    template <typename T, typename U>
    void initialize(const std::initializer_list<U>& init_list);
public:
    explicit Tensor(const std::vector<std::size_t>& shape, Forge::Dtype dtype = Forge::Dtype::float32, bool need_grads = true,
        Forge::Device device = Forge::Device::CPU);

    template <TensorStorageType T>
    Forge::Tensor& operator=(init_list_1D<T>);
    template <TensorStorageType T>
    Forge::Tensor& operator=(init_list_2D<T>);
    template <TensorStorageType T>
    Forge::Tensor& operator=(init_list_3D<T>);
    template <TensorStorageType T>
    Forge::Tensor& operator=(init_list_4D<T>);

    template <typename T>
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> as_eigen() const;

    static Dispatcher<UtilityOps>& dispatcher();

    static Tensor Constant(const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads=true,
        Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
    static Tensor Ones(const std::vector<std::size_t>& shape, bool need_grads=true,
    Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
    static Tensor Zeros(const std::vector<std::size_t>& shape, bool need_grads=true,
    Dtype dtype=Dtype::float32 ,Device device=Device::CPU);

    [[nodiscard]] const auto& need_grads() const {return m_need_grads;} ;
    [[nodiscard]] const auto& shape() const {return m_shape;}
    [[nodiscard]] const auto& strides() const {return m_strides;} ;
    [[nodiscard]] const auto& dtype() const {return m_dtype;} ;
    [[nodiscard]] const auto& device() const {return m_device;} ;
    [[nodiscard]] void* data() const {return m_storage->data();}
    [[nodiscard]] auto size() const {return m_size;}
    [[nodiscard]] auto dispatch_key() const {return m_dispatch_key;}
};

template <TensorStorageType T>
constexpr Forge::Dtype Forge::dtype_of() {
    if constexpr (std::is_same_v<T, std::int16_t>) return Dtype::int16;
    else if constexpr (std::is_same_v<T, std::int32_t>) return Dtype::int32;
    else if constexpr (std::is_same_v<T, std::int64_t>) return Dtype::int64;
    else if constexpr (std::is_same_v<T, Eigen::half>) return Dtype::float16;
    else if constexpr (std::is_same_v<T, float>) return Dtype::float32;
    else if constexpr (std::is_same_v<T, double>) return Dtype::float64;
    else static_assert(dependent_false_v<T>, " Unsupported tensor storage type for dtype_of()");
}

template<typename T>
void Forge::infer_shape(const T&, const std::vector<std::size_t>&){}
template<typename T>
void Forge::infer_shape(const std::initializer_list<T>& init_list, std::vector<std::size_t>& shape) {
    shape.push_back(init_list.size());
    infer_shape(*init_list.begin(), shape);
}
template <typename T>
void Forge::check_shape_and_throw(const std::initializer_list<T>& init_list, const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> list_shape{};
    if (infer_shape(init_list, list_shape); list_shape!=shape)
        throw std::invalid_argument("initializer list shape doesnt match with of tensor's");
}

template <typename T, typename U>
void Forge::flatten(const T& item, std::vector<U>& result) {
    result.push_back(item);
}
template <typename T, typename U>
void Forge::flatten(const std::initializer_list<T>& list, std::vector<U>& result) {
    for (const auto& element : list) {
        flatten(element, result);
    }
}
template<typename T, typename U>
std::vector<T> Forge::flatten_list(const std::initializer_list<U>& list) {
    std::vector<T> result;
    flatten(list, result);
    return result;
}

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


inline Forge::Tensor Forge::Tensor::Constant(const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads,
    Dtype dtype,Device device) {
    Tensor tensor(shape, dtype, need_grads, device);
    auto* constAbstract {dispatcher().lookup<ConstantAbstract>(UtilityOps::constant, tensor.m_dispatch_key)};
    constAbstract->setConstant(tensor.data(), tensor.size(), constant, dtype);
    return tensor;
}

inline Forge::Tensor Forge::Tensor::Ones(const std::vector<std::size_t>& shape, bool need_grads,
    Dtype dtype, Device device) {
    return Constant(shape, 1, need_grads, dtype, device);
}

inline Forge::Tensor Forge::Tensor::Zeros(const std::vector<std::size_t>& shape, bool need_grads,
    Dtype dtype, Device device) {
    return Constant(shape, 0, need_grads, dtype, device);
}

template<typename T, typename U>
void Forge::Tensor::initialize(const std::initializer_list<U> &init_list) {
    check_shape_and_throw(init_list, m_shape);
    auto flattened_list {flatten_list<T>(init_list)};
    const auto* initializer {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers, m_dispatch_key)};
    initializer->initialize(flattened_list.data(), m_storage->data(), dtype_of<T>(), m_dtype, m_size);
}


template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_1D<T> init_list) {initialize(init_list);return *this;}
template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_2D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_3D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_4D<T> init_list) {initialize<T>(init_list); return *this;}


template <typename T>
inline Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> Forge::Tensor::as_eigen() const {
    Eigen::array<Eigen::Index, 4> eigen_dims;
    eigen_dims.fill(1);
    for (int i{static_cast<int>(m_shape.size()-1)}; i>=0; i--) eigen_dims[i] = m_shape[i];
    return Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(static_cast<T*>(m_storage->data()), eigen_dims);
}


inline std::ostream& operator<<(std::ostream& os, const Forge::Tensor& tensor) {
    const auto* print {Forge::Tensor::dispatcher().lookup<Forge::PrintAbstract>(Forge::print, tensor.dispatch_key())};
    print->print(tensor.data(), tensor.shape(), tensor.strides(), tensor.dtype(), os);
    return os;
}