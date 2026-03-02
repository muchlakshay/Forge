#pragma once
#include <random>
#include "ops/Dispatcher/kernel_base.h"
#include "tensor.h"

namespace Forge {
    enum class Initializers{xavier_normal, xavier_uniform, he_normal, he_uniform};
    struct WeightsInitAbstract;
    struct WeightsInitCPU;
}

struct Forge::WeightsInitAbstract : Kernel {
    WeightsInitAbstract() : Kernel{ctti::type_id<WeightsInitAbstract>()} {}
    virtual void initialize(Tensor& weights, std::size_t input_size, std::size_t output_size,
        Initializers initializer) const = 0;
};

struct Forge::WeightsInitCPU : WeightsInitAbstract {
    void initialize(Tensor& weights, std::size_t input_size, std::size_t output_size,
        Initializers initializer) const override {
        std::random_device device;
        std::mt19937_64 mt{ device() };

        auto fill {
            [&](auto& storage, auto& dist) {
                DISPATCH_ALL_TYPES(weights.dtype(), Device::CPU,
                    [&] {
                        auto* casted_ptr {static_cast<scalar_t*>(weights.data())};
                        for (std::size_t i {}; i<weights.size(); ++i)
                            casted_ptr[i] = static_cast<scalar_t>(dist(mt));
                    });}
        };
    }
};

