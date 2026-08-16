#pragma once
#include "weights_init.h"

struct Forge::WeightsInitCPU : WeightsInitAbstract {
    void initialize(Tensor& weights, Initializers initializer)const override ;};
