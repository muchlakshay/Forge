#pragma once
#include "primitive_dispatcher.h"
#include "Storage/storage_abstract.h"

namespace Forge {
    enum class Initializers{xavier_normal, xavier_uniform, he_normal, he_uniform};
    struct WeightsInitAbstract;
    struct WeightsInitCPU;
}

struct Forge::WeightsInitAbstract : Kernel {
    WeightsInitAbstract() : Kernel{ctti::type_id<WeightsInitAbstract>()} {}
    virtual void initialize(std::shared_ptr<StorageAbstract>& weights, Initializers initializer) const = 0;
};

