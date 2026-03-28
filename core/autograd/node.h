#pragma once
#include "autograd/node_abstract.h"
#include "tensor/tensor.h"
#include <type_traits>

namespace Forge {
    template <typename T, typename U>
    concept isRefTo = std::is_reference_v<U> && std::is_same_v<std::remove_reference_t<U>, T>;

    template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
    class Node;
}
template <typename OpClass, std::size_t Parents, ValidEnum OpEnum>
class Forge::Node final : public NodeAbstract {
    std::array<Tensor, Parents> m_parents {};
    Tensor m_child{};
    DispatchKey m_dispatch_key {};
    const Dispatcher<OpEnum>& m_dispatcher {};
    OpEnum m_op {};
    const OpClass* m_gradFn{};
public:
    template <typename... ParentTensors>
    explicit Node(Device device, const Dispatcher<OpEnum>& dispatcher,OpEnum op, Tensor& child, ParentTensors&&... parent_tensors) :
    m_parents{Forge::Tensor{parent_tensors, Forge::NodeContext{}}...}, m_child{child, Forge::NodeContext{}},
    m_dispatch_key{device==Device::CPU?DispatchKey::CPU_Autodiff:DispatchKey::CUDA_Autodiff},
    m_dispatcher{dispatcher}, m_op{op}, m_gradFn{dispatcher.template lookup<OpClass>(op, m_dispatch_key)}{
        static_assert(sizeof...(parent_tensors)==Parents, "Invalid number of parent tensors passed");
    }
        void backward() const override;
};

template<typename OpClass, std::size_t Parents, ValidEnum OpEnum>
void Forge::Node<OpClass, Parents, OpEnum>::backward() const {
    [&]<std::size_t...Is>(std::index_sequence<Is...>) {
        m_gradFn->compute_grads(m_parents[Is]..., m_child);
    }(std::make_index_sequence<Parents>{});
    for (auto& parent : m_parents) {if (parent.node()) parent.node()->backward();}
}