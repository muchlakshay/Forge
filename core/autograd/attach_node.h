#pragma once
#include <autograd/node.h>
#include <tensor/tensor.h>

namespace Forge {
    template <typename OpClass, std::size_t Parents, ValidEnum OpEnum, typename... ParentTensors>
    void attach_node(Tensor& tensor, Device device, const Dispatcher<OpEnum>& dispatcher, OpEnum op, ParentTensors&&... parent_tensors);
}
