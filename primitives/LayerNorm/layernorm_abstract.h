#include "ops/Dispatcher/kernel_base.h"

namespace Forge {
    struct LayerNormImplAbstract;
    class Tensor;
}

struct Forge::LayerNormImplAbstract: Kernel {
    LayerNormImplAbstract() : Kernel{ctti::type_id<LayerNormImplAbstract>()} {}
    virtual void forward(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor& opt) const = 0;
};