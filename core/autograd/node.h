#pragma once
#include "autograd/node_abstract.h"
#include "tensor/tensor.h"

namespace Forge {
    template <typename OpClass, std::size_t Parents>
    class Node;
}
template <typename OpClass, std::size_t Parents>
class Forge::Node final : public NodeAbstract {
    std::array<std::shared_ptr<Tensor>, Parents> m_parents;
    DispatchKey m_dispatch_key {};
    public:
    explicit Node(Device device) : m_dispatch_key{device==Device::CPU?DispatchKey::CPU_Autodiff:DispatchKey::CUDA_Autodiff} {}
        void backward() const override;
};

