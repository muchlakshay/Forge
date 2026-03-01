#pragma once
#include <random>
#include "primitive_dispatcher.h"
#include "Storage/storage_abstract.h"

namespace Forge {
    enum class Initializers{xavier_normal, xavier_uniform, he_normal, he_uniform};
    struct WeightsInitAbstract;
    struct WeightsInitCPU;
}

struct Forge::WeightsInitAbstract : Kernel {
    WeightsInitAbstract() : Kernel{ctti::type_id<WeightsInitAbstract>()} {}
    virtual void initialize(std::shared_ptr<StorageAbstract>& weights, std::size_t input_size, std::size_t output_size,
        Initializers initializer) const = 0;
};

struct Forge::WeightsInitCPU : WeightsInitAbstract {
    void initialize(std::shared_ptr<StorageAbstract>& weights, std::size_t input_size, std::size_t output_size,
        Initializers initializer) const override {
        std::random_device device;
        std::mt19937_64 mt{ device() };

        auto fill {
            [&](auto& storage, auto& dist) {
                for (std::size_t i {}; i<storage->size(); ++i) storage[i] = dist(mt);
            }
        };

    }
};
