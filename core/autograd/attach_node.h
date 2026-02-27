#pragma once
#include <autograd/node.h>
#include <tensor/tensor.h>

namespace Forge {
    template <typename OpClass, std::size_t Parents, ValidEnum OpEnum, typename... ParentTensors>
    void attach_node(Tensor& tensor, Device device, const Dispatcher<OpEnum>& dispatcher, OpEnum op, ParentTensors&&... parent_tensors);
}


template<typename OpClass, std::size_t Parents, ValidEnum OpEnum, typename... ParentTensors>
void Forge::attach_node(Tensor &tensor, Device device, const Dispatcher<OpEnum> & dispatcher,
    OpEnum op, ParentTensors &&... parent_tensors) {
    auto node {std::make_shared<Node<OpClass, Parents, OpEnum>>(device, dispatcher, op, tensor,
        std::forward<ParentTensors>(parent_tensors)...)};
    tensor.node() = node;
}