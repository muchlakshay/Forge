#pragma once
#include "tensor/tensor.h"
#include "primitive_dispatcher.h"
#include "weights_init.h"

namespace Forge {
    class Linear;
    struct LinearAbstract;
    struct LinearCPU;
}

class Forge::Linear {
    Tensor m_weights;
    Tensor m_bias;
    DispatchKey m_dispatch_key;
    std::size_t m_input_size, m_output_size;
    bool m_using_bias;
    Dtype m_dtype;
    Device m_device;
public:
    Linear(std::size_t input_size, std::size_t output_size, Initializers initializer=Initializers::xavier_normal,
        Dtype dtype=Dtype::float32, Device device=Device::CPU, bool bias=true);
};

