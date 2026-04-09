#pragma once
#include "lossFunctionsGradsCPU.h"
#include "../lossFunctionsGrads.h"

namespace Forge {
    struct MSEGradsCPU;
    struct CrossEntropyGradsCPU;
    struct BinaryCrossEntropyGradsCPU;
}

struct Forge::MSEGradsCPU : MSEGradsAbstract {void compute_grads(const Tensor &pred, const Tensor &ground_truth, const Tensor& loss) const override;};
struct Forge::CrossEntropyGradsCPU : CrossEntropyGradsAbstract {void compute_grads(const Tensor &pred, const Tensor &ground_truth, const Tensor& loss) const override;};
struct Forge::BinaryCrossEntropyGradsCPU : BinaryCrossEntropyGradsAbstract {void compute_grads(const Tensor &pred, const Tensor &ground_truth, const Tensor& loss) const override;};
