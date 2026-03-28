#pragma once
#include "../primitives/weights_init.h"

struct Forge::WeightsInitCPU : WeightsInitAbstract {
    void initialize(Tensor& weights, std::size_t input_size, std::size_t output_size,
        Initializers initializer)const override ;};
