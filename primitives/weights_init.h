#pragma once
#include "primitive_dispatcher.h"
#include "Storage/storage_abstract.h"

namespace Forge {
    struct WeightsInitAbstract;
    struct WeightsInitCPU;
}

struct Forge::WeightsInitAbstract : Kernel {
    WeightsInitAbstract() : Kernel{ctti::type_id<WeightsInitAbstract>()} {}
    virtual initialize(std::shared_ptr<StorageAbstract>& weights) const = 0;
};


