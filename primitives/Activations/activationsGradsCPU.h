#pragma once
#include "acvitvationsGradsAbstract.h"

namespace Forge {
    struct ReluGradsCPU;
    struct LeakyReluGradsCPU;
    struct SoftmaxGradsCPU;
    struct GeluGradsCPU;
    struct SigmoidGradsCPU;
    struct TanhGradsCPU;
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

struct Forge::GeluGradsCPU : GeluGradsAbstract {
  void compute_grads(const Tensor& preactivations, const Tensor& activations) const override {
      DISPATCH_ALL_TYPES(preactivations.dtype(), Device::CPU, [&] {
          auto preact_map {preactivations.as_eigen<scalar_t>()};
          auto preact_grads_map {preactivations.gradients().as_eigen<scalar_t>()};
          auto act_grads_map {activations.gradients().as_eigen<scalar_t>()};

          const auto k {static_cast<scalar_t>(0.7978845608028654)}, c {static_cast<scalar_t>(0.044715)};
          const auto half {static_cast<scalar_t>(0.5)}, one {static_cast<scalar_t>(1)}, three{static_cast<scalar_t>(3)};

          const auto preact_sqr {preact_map.square()};
          const auto preact_cube {preact_sqr*preact_map};
          const auto u {k*(preact_map+c*preact_cube)};
          const auto tanh_u {u.tanh()};

          const auto term1 {half*(one+tanh_u)};
          const auto sech2 {one-tanh_u.square()};
          const auto term2 {half*preact_map*sech2*k*(one+three*c*preact_sqr)};

          preact_grads_map += act_grads_map*(term1+term2);
        });
    }
};

struct Forge::SoftmaxGradsCPU : SoftmaxGradsAbstract {
    void compute_grads(const Tensor& logits, const Tensor& activations) const override {
        DISPATCH_ALL_TYPES(logits.dtype(), Device::CPU, [&] {
            auto logits_grads_map {logits.gradients().as_eigen<scalar_t>()};
            auto act_map {activations.as_eigen<scalar_t>()};
            auto act_grads_map {activations.gradients().as_eigen<scalar_t>()};

            auto act_mul_act_grads {act_grads_map*act_map};
            Eigen::array<Eigen::Index, 1> reduce_axis {3};
            auto dims {act_map.dimensions()};
            auto dot {act_mul_act_grads.sum(reduce_axis)};
            Eigen::array<Eigen::Index, 4> reshape_dims {dims[0], dims[1], dims[2], 1};

            auto dot_reshaped {dot.reshape(reshape_dims)};
            Eigen::array<Eigen::Index, 4> broadcast_dims {1, 1, 1, dims[3]};
            logits_grads_map += act_map * (act_grads_map-dot_reshaped.broadcast(broadcast_dims));
        });
    }
};

struct Forge::SigmoidGradsCPU : SigmoidGradsAbstract {
    void compute_grads(const Tensor& preactivations, const Tensor& activations) const override {
        DISPATCH_ALL_TYPES(preactivations.dtype(), Device::CPU, [&] {
            auto preact_grads_map {preactivations.gradients().as_eigen<scalar_t>()};
            auto act_grads_map {activations.gradients().as_eigen<scalar_t>()};
            auto act_map{activations.as_eigen<scalar_t>()};

            preact_grads_map += act_grads_map * (act_map*(static_cast<scalar_t>(1)-act_map));
        });
    }
};

struct Forge::TanhGradsCPU : TanhGradsAbstract {
    void compute_grads(const Tensor& preactivations, const Tensor& activations) const override {
        DISPATCH_ALL_TYPES(preactivations.dtype(), Device::CPU, [&] {
            auto preact_grads_map {preactivations.gradients().as_eigen<scalar_t>()};
            auto act_grads_map {activations.gradients().as_eigen<scalar_t>()};
            auto act_map{activations.as_eigen<scalar_t>()};
            preact_grads_map += act_grads_map * (static_cast<scalar_t>(1)-act_map.square());
        });
    }
};