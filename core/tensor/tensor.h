#pragma once
#include "Storage/storage_cpu.h"
#include "enums.h"
#include "../ops/Dispatcher/dispatcher.h"
#include "../ops/Ops Kernels/kernels_registrar.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <ranges>
#include <memory>
#include "autograd/node_abstract.h"

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

    struct NodeContext{};
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
    std::shared_ptr<NodeAbstract> m_node{};
    std::shared_ptr<Tensor> m_grads{};

    template <typename T, typename U>
    void initialize(const std::initializer_list<U>& init_list);
public:
    Tensor() = default;
    explicit Tensor(const std::vector<std::size_t>& shape, Forge::Dtype dtype = Forge::Dtype::float32, bool need_grads = true,
        Forge::Device device = Forge::Device::CPU);
    Tensor(const Tensor& another, struct NodeContext);
    Tensor(const Tensor& another);
    Tensor& operator=(const Tensor& another);

    template <TensorStorageType T>
    Tensor& operator=(init_list_1D<T>);
    template <TensorStorageType T>
    Tensor& operator=(init_list_2D<T>);
    template <TensorStorageType T>
    Tensor& operator=(init_list_3D<T>);
    template <TensorStorageType T>
    Tensor& operator=(init_list_4D<T>);
    template <TensorStorageType T>
    Tensor& operator=(T);

    Tensor operator[](std::size_t index);

    template <typename T>
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> as_eigen() const;

    static Dispatcher<UtilityOps>& dispatcher();

    template<TensorStorageType T>
    void setConstant(T constant);

    static Tensor Constant(const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads=true,
        Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
    static Tensor Ones(const std::vector<std::size_t>& shape, bool need_grads=true,
    Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
    static Tensor Zeros(const std::vector<std::size_t>& shape, bool need_grads=true,
    Dtype dtype=Dtype::float32 ,Device device=Device::CPU);
    template <TensorStorageType T>
    static Tensor FromHostPtr(T* host_ptr, const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads=true,
    Dtype dtype=Dtype::float32);
    template <typename T>
    static Tensor Range(T start, T end, T step=1, bool need_grads=true, Device device=Device::CPU);

    [[nodiscard]] const auto& need_grads() const {return m_need_grads;} ;
    [[nodiscard]] const auto& shape() const {return m_shape;}
    [[nodiscard]] const auto& strides() const {return m_strides;} ;
    [[nodiscard]] const auto& dtype() const {return m_dtype;} ;
    [[nodiscard]] const auto& device() const {return m_device;} ;
    [[nodiscard]] void* data() const {return m_storage->data();}
    [[nodiscard]] auto size() const {return m_size;}
    [[nodiscard]] auto dispatch_key() const {return m_dispatch_key;}

    template <typename... Dims> requires ((std::is_integral_v<Dims> && (!std::is_same_v<bool, Dims>)) && ...)
    Tensor reshape(Dims... dims);
    [[nodiscard]] Tensor clone() const;
    [[nodiscard]] auto& node() const {return m_node;}
    [[nodiscard]] auto& node() {return m_node;}

    [[nodiscard]] auto& grads() const {return m_grads;}
    [[nodiscard]] auto& gradients() const {return *(grads());}
    void backward() const {if (m_node) {grads()->setConstant(1.0f), m_node->backward();}}
    void clear_grads() const {if (m_grads) m_grads->setConstant(0.f);}

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
        if (shape[i]==0) throw std::invalid_argument("tensor shape must be non-zero");
        m_strides[i] = size;
        size*=m_shape[i];
    }
    m_size = size;
    static const auto storage_backend { dispatcher().lookup<StorageBackend>(UtilityOps::storage_backend, m_dispatch_key)};
    storage_backend->getStorageBackend(m_storage, size, m_dtype);
    if (need_grads) {
        m_grads = std::make_shared<Tensor>(m_shape, Dtype::float32, false, m_device);
        grads()->setConstant(0.f);
    }
}

inline Forge::Tensor::Tensor(const Tensor &another) : m_device{another.m_device}, m_dtype{another.m_dtype},
        m_storage {another.m_storage}, m_shape{another.m_shape}, m_strides {another.m_strides},
        m_need_grads {another.m_need_grads}, m_dispatch_key{another.m_dispatch_key} {
    if (m_need_grads) {
        m_grads = std::make_shared<Tensor>(m_shape, Dtype::float32, false, m_device);
        grads()->setConstant(0.f);
    }
}

inline Forge::Tensor& Forge::Tensor::operator=(const Tensor& another) {
    Tensor copy{another};
    *this = std::move(copy);
    return *this;
}


inline Forge::Tensor::Tensor(const Tensor &another, NodeContext) : m_device{another.m_device}, m_dtype{another.m_dtype},
        m_storage {another.m_storage}, m_shape{another.m_shape}, m_strides {another.m_strides},
        m_need_grads {another.m_need_grads}, m_dispatch_key{another.m_dispatch_key}, m_node{another.node()} {
    m_grads = another.grads();
}

inline Forge::Tensor Forge::Tensor::Constant(const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads,
    Dtype dtype,Device device) {
    Tensor tensor(shape, dtype, need_grads, device);
    static const auto* constAbstract {dispatcher().lookup<ConstantAbstract>(UtilityOps::constant, tensor.m_dispatch_key)};
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

template<typename T>
Forge::Tensor Forge::Tensor::Range(T start, T end, T step, bool need_grads, Device device) {
    T value {start};
    std::vector<T> range{};
    for (;value<end; value+=step) range.push_back(value);
    Tensor range_tensor {{range.size()}, dtype_of<T>(), need_grads, device};
    static const auto* filler {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers,
        range_tensor.dispatch_key())};
    filler->initialize(&range.front(), range_tensor.m_storage->data(), range_tensor.m_dtype,
        range_tensor.m_dtype, range.size());
    return range_tensor;
}


template <TensorStorageType T>
Forge::Tensor& Forge::Tensor::operator=(T val){
    static const auto* initializer {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers, m_dispatch_key)};
    initializer->initialize(&val, m_storage->data(), dtype_of<T>(), m_dtype, 1);
    return *this;
}

template<typename T, typename U>
void Forge::Tensor::initialize(const std::initializer_list<U> &init_list) {
    check_shape_and_throw(init_list, m_shape);
    auto flattened_list {flatten_list<T>(init_list)};
    static const auto* initializer {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers, m_dispatch_key)};
    initializer->initialize(flattened_list.data(), m_storage->data(), dtype_of<T>(), m_dtype, m_size);
}

template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_1D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_2D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_3D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
inline Forge::Tensor& Forge::Tensor::operator=(init_list_4D<T> init_list) {initialize<T>(init_list); return *this;}

inline Forge::Tensor Forge::Tensor::operator[](std::size_t index) {
    if (m_shape.empty() || index>=m_shape.front()) throw std::out_of_range(std::format("Index {} is out of bounds", index));
    std::vector<std::size_t> shape (m_shape.size()-1), strides (m_strides.size()-1);
    std::copy(m_shape.begin()+1, m_shape.end(), shape.begin());
    std::copy(m_strides.begin()+1, m_strides.end(), strides.begin());

    Tensor view {};
    view.m_shape = shape;
    view.m_strides = strides;
    view.m_device = m_device;
    view.m_need_grads = m_need_grads;
    view.m_dtype = m_dtype;
    view.m_dispatch_key = m_dispatch_key;
    view.m_size = shape.empty() ? 1
    : std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>());
    static const auto* storageBackend {dispatcher().lookup<StorageBackend>(UtilityOps::storage_backend, m_dispatch_key)};
    storageBackend->setView(m_storage, view.m_storage, m_strides.front()*index, view.m_size, view.m_dtype);
    return view;
}

template <typename T>
Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> Forge::Tensor::as_eigen() const {
    Eigen::array<Eigen::Index, 4> eigen_dims;
    eigen_dims.fill(1);
    for (int i{static_cast<int>(m_shape.size()-1)}; i>=0; i--) eigen_dims[i] = m_shape[i];
    return Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>(static_cast<T*>(m_storage->data()), eigen_dims);
}


template <typename... Dims> requires ((std::is_integral_v<Dims> && (!std::is_same_v<bool, Dims>)) && ...)
Forge::Tensor Forge::Tensor::reshape(Dims... dims) {
    auto size {(1*...*dims)};
    if (size!=m_size) throw std::invalid_argument(
        std::format("Invalid Dim Passed: size {} doesnt match to the original {}", size, m_size));
    size = 1;
    if (sizeof...(dims)>4) throw std::invalid_argument("Reshaped tensor rank must be at max 4");

    std::vector<std::size_t> new_shape {dims...}, new_strides (new_shape.size());
    for (int i = new_shape.size()-1; i>=0; --i) {
        new_strides[i] = size;
        size*=new_shape[i];
    }
    Tensor reshaped {*this};
    reshaped.m_shape = std::move(new_shape);
    reshaped.m_strides = std::move(new_strides);
    return reshaped;
}


template<TensorStorageType T>
void Forge::Tensor::setConstant(T constant) {
    static const auto* filler {dispatcher().lookup<ConstantAbstract>(UtilityOps::constant, m_dispatch_key)};
    filler->setConstant(m_storage->data(), m_size, constant, m_dtype);
}

inline Forge::Tensor Forge::Tensor::clone() const {
    Tensor clone {*this};
    static const auto* StorageCpy {dispatcher().lookup<StorageCopyAbstract>(UtilityOps::storage_copy,
        m_dispatch_key)};
    StorageCpy->copy_storage(m_storage, clone.m_storage, m_dtype);
    return clone;
}

inline std::ostream& operator<<(std::ostream& os, const Forge::Tensor& tensor) {
    static const auto* print {Forge::Tensor::dispatcher().lookup<Forge::PrintAbstract>(Forge::UtilityOps::print, tensor.dispatch_key())};
    print->print(tensor.data(), tensor.shape(), tensor.strides(), tensor.dtype(), os);
    return os;
}