#pragma once
#include "lossFunctionsCPU.h"
#include "../lossFunctionsAbstract.h"

namespace Forge {
    struct MSE_CPU;
    struct CrossEntropyCPU;
    struct BinaryCrossEntropyCPU;
}

struct Forge::MSE_CPU final : MSEAbstract {void compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const override;};
struct Forge::CrossEntropyCPU final : CrossEntropyAbstract {void compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const override;};
struct Forge::BinaryCrossEntropyCPU final : BinaryCrossEntropyAbstract{void compute_loss(const Tensor &pred, const Tensor &ground_truth, Tensor &loss) const override;};