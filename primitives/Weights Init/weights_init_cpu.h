#pragma once
#include "weights_init.h"

struct Forge::WeightsInitCPU : WeightsInitAbstract {
    void initialize(Tensor& weights, const std::vector<std::size_t>& dims,
        Initializers initializer)const override ;};
