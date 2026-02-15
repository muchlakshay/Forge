#pragma once
#include "autograd/node_abstract.h"
#include "tensor/tensor.h"
#include <type_traits>

namespace Forge {
    template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
    class Node;
}
template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
class Forge::Node final : public NodeAbstract {
    std::array<std::shared_ptr<Tensor>, Parents> m_parents {};
    std::shared_ptr<Tensor> m_child{};
    DispatchKey m_dispatch_key {};
    const Dispatcher<OpEnum>& m_dispatcher {};
    OpEnum m_op {};
public:
    template <typename... ParentTensors>
    explicit Node(Device device, const Dispatcher<OpEnum>& dispatcher,OpEnum op, ParentTensors&&... parent_tensors) :
    m_parents{parent_tensors...}, m_dispatch_key{device==Device::CPU?DispatchKey::CPU_Autodiff:DispatchKey::CUDA_Autodiff},
    m_dispatcher{dispatcher}, m_op{op}{
        static_assert(sizeof...(parent_tensors)<=Parents, "Invalid number of parent tensors passed");
    }
        void backward() const override;
};
