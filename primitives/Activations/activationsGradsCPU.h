#pragma once
#include "acvitvationsGradsAbstract.h"

namespace Forge {
    struct ReluGradsCPU;
    struct SigmoidCPU;
    struct LeakyReluGradsCPU;
    struct SoftmaxGradsCPU;
    struct GeluGradsCPU;
    struct SigmoidGradsCPU;
}

struct Forge::ReluGradsCPU : ReluGradsAbstract {
    void compute_grads(const Tensor& preactivations, const Tensor& activations) const override {
        DISPATCH_ALL_TYPES(preactivations.dtype(), Device::CPU, [&] {
            auto preact_map {preactivations.as_eigen<scalar_t>()};
            auto preact_grads_map {preactivations.gradients().as_eigen<scalar_t>()};
            auto act_grads_map {activations.gradients().as_eigen<scalar_t>()};
            auto zero {static_cast<scalar_t>(0)};
            preact_grads_map += act_grads_map*(preact_map>zero).select(act_grads_map, zero);
        });
    }
};

struct Forge::LeakyReluGradsCPU : LeakyReluGradsAbstract {
  void compute_grads(const Tensor& preactivations, const Tensor& activations) const override {
      DISPATCH_ALL_TYPES(preactivations.dtype(), Device::CPU, [&] {
          auto preact_map {preactivations.as_eigen<scalar_t>()};
          auto preact_grads_map {preactivations.gradients().as_eigen<scalar_t>()};
          auto act_grads_map {activations.gradients().as_eigen<scalar_t>()};
          const auto alpha {static_cast<scalar_t>(0.01)}, zero{static_cast<scalar_t>(0)}, one {static_cast<scalar_t>(1)};
          preact_grads_map += act_grads_map*(preact_map>zero).select(one, alpha);
      });
  }
};

