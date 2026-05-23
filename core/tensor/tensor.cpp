#include "tensor.h"

#include "autograd/attach_node.h"

Forge::Tensor::Tensor(const std::vector<std::size_t>& shape, Forge::Dtype dtype, bool need_grads,
                      Forge::Device device) : m_device{device}, m_dtype{dtype}, m_shape{shape}, m_strides(shape.size(), 0), m_need_grads{need_grads},
                                              m_dispatch_key {device==Device::CPU?DispatchKey::CPU : DispatchKey::CUDA}{

    std::size_t size {1u};
    for (int i = shape.size()-1; i>=0; i--) {
        if (shape[i]==0) throw std::invalid_argument("tensor shape must be non-zero");
        m_strides[i] = size;
        size*=m_shape[i];
    }
    m_size = size;
    const auto* storage_backend { dispatcher().lookup<StorageBackend>(UtilityOps::storage_backend, m_dispatch_key)};
    storage_backend->getStorageBackend(m_storage, size, m_dtype);
    if (need_grads) {
        m_grads = std::make_shared<Tensor>(m_shape, Dtype::float32, false, m_device);
        m_grads->setConstant(0.f);
    }
}

Forge::Tensor::Tensor(const Tensor &another) : m_device{another.m_device}, m_dtype{another.m_dtype},
        m_storage {another.m_storage}, m_shape{another.m_shape}, m_strides {another.m_strides},
        m_need_grads {another.m_need_grads}, m_dispatch_key{another.m_dispatch_key}, m_size {another.m_size} {
    if (m_need_grads) {
        m_grads = std::make_shared<Tensor>(m_shape, Dtype::float32, false, m_device);
        grads()->setConstant(0.f);
    }
}

Forge::Tensor& Forge::Tensor::operator=(const Tensor& another) {
    Tensor copy{another};
    *this = std::move(copy);
    return *this;
}

Forge::Tensor::Tensor(const Tensor &another, NodeContext) : m_device{another.m_device}, m_dtype{another.m_dtype},
        m_storage {another.m_storage}, m_shape{another.m_shape}, m_strides {another.m_strides}, m_size {another.m_size},
        m_need_grads {another.m_need_grads}, m_dispatch_key{another.m_dispatch_key}, m_node{another.node()} {
    m_grads = another.grads();
}

Forge::Tensor Forge::Tensor::Constant(const std::vector<std::size_t>& shape, const Scalar& constant, bool need_grads,
    Dtype dtype,Device device) {
    Tensor tensor(shape, dtype, need_grads, device);
    const auto* constAbstract {dispatcher().lookup<ConstantAbstract>(UtilityOps::constant, tensor.m_dispatch_key)};
    constAbstract->setConstant(tensor.data(), tensor.size(), constant, dtype);
    return tensor;
}

Forge::Tensor Forge::Tensor::Ones(const std::vector<std::size_t>& shape, bool need_grads,
    Dtype dtype, Device device) {
    return Constant(shape, 1, need_grads, dtype, device);
}

Forge::Tensor Forge::Tensor::Zeros(const std::vector<std::size_t>& shape, bool need_grads,
    Dtype dtype, Device device) {
    return Constant(shape, 0, need_grads, dtype, device);
}

Forge::Dispatcher<Forge::UtilityOps>& Forge::Tensor::dispatcher() {
    static Dispatcher<UtilityOps> utility_dispatcher;
    static bool is_defined = false;
    if (is_defined) return utility_dispatcher;
    Forge::register_all_kernels(utility_dispatcher);
    is_defined = true;
    return utility_dispatcher;
}

Forge::Tensor Forge::Tensor::operator[](std::size_t index) {
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
    const auto* storageBackend {dispatcher().lookup<StorageBackend>(UtilityOps::storage_backend, m_dispatch_key)};
    storageBackend->setView(m_storage, view.m_storage, m_strides.front()*index, view.m_size, view.m_dtype);
    return view;
}

Forge::Tensor Forge::Tensor::clone() const {
    Tensor clone {*this};
    const auto* StorageCpy {dispatcher().lookup<StorageCopyAbstract>(UtilityOps::storage_copy,
        m_dispatch_key)};
    StorageCpy->copy_storage(m_storage, clone.m_storage, m_dtype);
    return clone;
}

void Forge::Tensor::copy(const Tensor& another) {
    if (another.m_shape!=m_shape) throw std::invalid_argument("argument Tensor shapes is different");
    if (another.m_device!=m_device) throw std::invalid_argument("argument Tensor resides on different device");
    if (another.m_dtype!=m_dtype) throw std::invalid_argument("argument Tensor have different dtype");
    const auto* StorageCpy {dispatcher().lookup<StorageCopyAbstract>(UtilityOps::storage_copy, m_dispatch_key)};
    StorageCpy->copy_storage(another.m_storage, m_storage, m_dtype);
}

Forge::Tensor Forge::Tensor::Random(const std::vector<std::size_t> &shape, Dtype dtype, Initializers initializer, Device device,
    bool need_grads) {
    Tensor opt {shape, dtype, need_grads, device};
    const auto* init {dispatcher().lookup<RandomAbstract>(UtilityOps::random, opt.dispatch_key())};
    init->randomize(opt, initializer);
    return opt;
}

Forge::Tensor Forge::Tensor::BroadcastAdd(const Tensor &another, const std::vector<int> &bcast_dims) {
    if (bcast_dims.empty()) throw std::invalid_argument("bcast_dims is empty");
    if (bcast_dims.size()>4) throw std::invalid_argument("bcast_dims size cant be more then 4");
    if (m_device!=another.m_device) throw std::invalid_argument("argument Tensor resides on different device");
    if (m_dtype != another.m_dtype) throw std::invalid_argument("argument Tensor have different dtype");

    Tensor opt {{m_shape}, m_dtype, m_need_grads || another.need_grads(), m_device};
    const auto* impl {dispatcher().lookup<BroadcastAddAbstract>(UtilityOps::bcast_add, m_dispatch_key)};
    impl->add(*this, another, bcast_dims, opt);
    if (opt.need_grads()) {
        auto bcast_dims_tensor {FromHostPtr(const_cast<int*>(bcast_dims.data()), {bcast_dims.size()}, false).clone()};
        attach_node<BroadcastAddGradsAbstract, 3>(opt, m_device, dispatcher(),
        UtilityOps::bcast_add, *this, another, bcast_dims_tensor);
    }
    return opt;
}
