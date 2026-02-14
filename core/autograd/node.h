#pragma once
#include "autograd/node_abstract.h"
#include "tensor/tensor.h"

namespace Forge {
    template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
    class Node;
}
template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
class Forge::Node final : public NodeAbstract {
    std::array<std::shared_ptr<Tensor>, Parents> m_parents;
    DispatchKey m_dispatch_key {};
    const Dispatcher<OpClass>& m_dispatcher {};
    public:
    explicit Node(Device device, const Dispatcher<OpEnum>& dispatcher) :
    m_dispatch_key{device==Device::CPU?DispatchKey::CPU_Autodiff:DispatchKey::CUDA_Autodiff}, m_dispatcher{dispatcher} {}
        void backward() const override;
};

