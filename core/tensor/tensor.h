#pragma once
#include "Storage/CPU/storage_cpu.h"
#include "enums.h"
#include "../ops/Dispatcher/dispatcher.h"
#include "../ops/Ops Kernels/kernels_registrar.h"
#include "autograd/node_abstract.h"
#include "execution_ctx.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <ranges>
#include <memory>
#include <iostream>

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
    mutable std::shared_ptr<NodeAbstract> m_node{};
    std::shared_ptr<Tensor> m_grads{};

    template <typename T, typename U>
    void initialize(const std::initializer_list<U>& init_list);
public:
    Tensor() = default;
    explicit Tensor(const std::vector<std::size_t>& shape, Dtype dtype = Dtype::float32, bool need_grads = true,
        Device device = Device::CPU);
    Tensor(const Tensor& another, NodeContext);
    Tensor(const Tensor& another);
    Tensor(Tensor&& another) = default;
    Tensor& operator=(const Tensor& another);
    Tensor& operator=(Tensor&& another) = default;

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

    template <typename T, std::size_t Rank=4>
    Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>> as_eigen() const;

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
    static Tensor FromHostPtr(T* host_ptr, const std::vector<std::size_t>& shape, bool need_grads=true);
    template <typename T>
    static Tensor Range(T start, T end, T step=1, bool need_grads=true, Device device=Device::CPU);
    static Tensor Random(const std::vector<std::size_t>& shape, Dtype=Dtype::float32,
        Initializers initializer=Initializers::xavier_normal, Device device=Device::CPU, bool need_grads=true);

    void copy(const Tensor& another);
    Tensor BroadcastAdd(const Tensor& another, const std::vector<int>& bcast_dims={1, 1, 1, 1});

    [[nodiscard]] const auto& need_grads() const {return m_need_grads;} ;
    [[nodiscard]] const auto& shape() const {return m_shape;}
    [[nodiscard]] const auto& strides() const {return m_strides;} ;
    [[nodiscard]] const auto& dtype() const {return m_dtype;} ;
    [[nodiscard]] const auto& device() const {return m_device;} ;
    [[nodiscard]] void* data() const {return m_storage?m_storage->data():nullptr;}
    [[nodiscard]] auto size() const {return m_size;}
    [[nodiscard]] auto dispatch_key() const {return m_dispatch_key;}
    [[nodiscard]] const auto& storage() const {return m_storage;}

    template <typename... Dims> requires ((std::is_integral_v<Dims> && (!std::is_same_v<bool, Dims>)) && ...)
    Tensor reshape(Dims... dims);
    [[nodiscard]] Tensor clone() const;
    [[nodiscard]] auto& node() const {return m_node;}
    [[nodiscard]] auto& node() {return m_node;}

    [[nodiscard]] auto& grads() const {return m_grads;}
    [[nodiscard]] auto& gradients() const {return *(grads());}
    void backward(const bool keep_graph=false, bool contain_upstream_grads=false) const {
        if (m_node) {
            if (!contain_upstream_grads) grads()->setConstant(1.0f); m_node->backward();
            if (!keep_graph) m_node.reset();
        }
    }
    void clear_grads() const {if (m_grads) m_grads->setConstant(0.f);}

};

template <TensorStorageType T>
constexpr Forge::Dtype Forge::dtype_of() {
    // if constexpr (std::is_same_v<T, std::int16_t>) return Dtype::int16;
    // else if constexpr (std::is_same_v<T, std::int32_t>) return Dtype::int32;
    // else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, long long>) return Dtype::int64;
    // else if constexpr (std::is_same_v<T, Eigen::half>) return Dtype::float16;
    // else if constexpr (std::is_same_v<T, float>) return Dtype::float32;
    // else if constexpr (std::is_same_v<T, double>) return Dtype::float64;
    // else if constexpr (std::is_same_v<T, native_fp16_t>) return Dtype::float16;
    // else static_assert(dependent_false_v<T>, " Unsupported tensor storage type for dtype_of()");

    if constexpr (std::is_same_v<T, std::int32_t>) return Dtype::int32;
    else if constexpr (std::is_same_v<T, float>) return Dtype::float32;
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

template<TensorStorageType T>
Forge::Tensor Forge::Tensor::FromHostPtr(T *host_ptr, const std::vector<std::size_t> &shape,
    bool need_grads) {
    if (!host_ptr) throw std::invalid_argument("host_ptr is null");
    Tensor tensor {};
    tensor.m_shape = shape; tensor.m_strides.resize(shape.size());
    std::size_t size {1u};
    for (auto i {static_cast<int>(shape.size())-1}; i>=0; i--) {
        if (shape[i]==0) throw std::invalid_argument("tensor shape must be non-zero");
        tensor.m_strides[i] = size;
        size*=tensor.m_shape[i];
    }
    tensor.m_size = size; tensor.m_dtype = dtype_of<std::remove_cvref_t<T>>(); tensor.m_need_grads = need_grads;
    tensor.m_storage = std::make_shared<StorageCPU<T>>(host_ptr, 0, size);
    return tensor;
}

template<typename T>
Forge::Tensor Forge::Tensor::Range(T start, T end, T step, bool need_grads, Device device) {
    T value {start};
    std::vector<T> range{};
    for (;value<end; value+=step) range.push_back(value);
    Tensor range_tensor {{range.size()}, dtype_of<T>(), need_grads, device};
    const auto* filler {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers,
        range_tensor.dispatch_key())};
    filler->initialize(&range.front(), range_tensor.m_storage->data(), range_tensor.m_dtype,
        range_tensor.m_dtype, range.size());
    return range_tensor;
}


template <TensorStorageType T>
Forge::Tensor& Forge::Tensor::operator=(T val){
    const auto* initializer {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers, m_dispatch_key)};
    initializer->initialize(&val, m_storage->data(), dtype_of<T>(), m_dtype, 1);
    return *this;
}

template<typename T, typename U>
void Forge::Tensor::initialize(const std::initializer_list<U> &init_list) {
    check_shape_and_throw(init_list, m_shape);
    auto flattened_list {flatten_list<T>(init_list)};
    const auto* initializer {dispatcher().lookup<InitializeAbstract>(UtilityOps::initializers, m_dispatch_key)};
    initializer->initialize(flattened_list.data(), m_storage->data(), dtype_of<T>(), m_dtype, m_size);
}

template<TensorStorageType T>
Forge::Tensor& Forge::Tensor::operator=(init_list_1D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
Forge::Tensor& Forge::Tensor::operator=(init_list_2D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
Forge::Tensor& Forge::Tensor::operator=(init_list_3D<T> init_list) {initialize<T>(init_list); return *this;}
template<TensorStorageType T>
Forge::Tensor& Forge::Tensor::operator=(init_list_4D<T> init_list) {initialize<T>(init_list); return *this;}

template <typename T, std::size_t Rank>
Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>> Forge::Tensor::as_eigen() const {
    if (Rank<m_shape.size()) throw std::invalid_argument(std::format("Rank must be at least equal to {}", m_shape.size()));
    if (Rank>4) throw std::invalid_argument("tensor rank can be 4 at max");

    Eigen::array<Eigen::Index, Rank> eigen_dims;
    eigen_dims.fill(1);
    for (int i = 0; i < m_shape.size(); i++) eigen_dims[Rank - m_shape.size() + i] = m_shape[i];
    return Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>>(static_cast<T*>(m_storage->data()), eigen_dims);
}


template <typename... Dims> requires ((std::is_integral_v<Dims> && (!std::is_same_v<bool, Dims>)) && ...)
Forge::Tensor Forge::Tensor::reshape(Dims... dims) {
    auto size {(1*...*dims)};
    if (size!=m_size) throw std::invalid_argument(
        std::format("Invalid Dim Passed: size {} doesnt match to the original {}", size, m_size));
    size = 1;
    if (sizeof...(dims)>4) throw std::invalid_argument("Reshaped tensor rank must be at max 4");

    std::vector<std::size_t> new_shape {dims...}, new_strides (new_shape.size());
    for (int i = static_cast<int>(new_shape.size()-1); i>=0; --i) {
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
    const auto* filler {dispatcher().lookup<ConstantAbstract>(UtilityOps::constant, m_dispatch_key)};
    // if (!(m_storage->data())) std::cout<<"null\n";
    filler->setConstant(m_storage->data(), m_size, constant, m_dtype);
}

inline std::ostream& operator<<(std::ostream& os, const Forge::Tensor& tensor) {
    const auto* print {Forge::Tensor::dispatcher().lookup<Forge::PrintAbstract>(Forge::UtilityOps::print, tensor.dispatch_key())};
    print->print(tensor.data(), tensor.shape(), tensor.strides(), tensor.dtype(), os);
    return os;
}

template <typename  T>
 std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    std::cout<<"[";
    for (auto& e : vec) std::cout<<e<<(&e!=&vec.back()?", ":"");
    std::cout<<"]"<<std::endl;
    return os;
}