#include "LinearLayerCPU.h"

void Forge::LinearCPU::forward(const Tensor& input, const Tensor& output, const Tensor& weights,
    const Tensor& bias, bool using_bias) const {
    Tensor weights_T {{weights.shape()[1], weights.shape()[0]}, weights.dtype(), false};
    Tensor bias_cast {output.shape(), bias.dtype(), false};
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto weights_map { weights.as_eigen<scalar_t>()};
        auto input_map   { input.as_eigen<scalar_t>()  };
        auto output_map  { output.as_eigen<scalar_t>() };
        auto weights_T_map {weights_T.as_eigen<scalar_t>()};

        Eigen::array<std::size_t , 4> shuffling {0, 1, 3, 2};
        forge_eval(weights_T_map, weights_map.shuffle(shuffling));

        Eigen::array<Eigen::IndexPair<std::size_t>, 1> contraction_dims {Eigen::IndexPair<std::size_t>(3, 2)};
        auto contract {input_map.contract(weights_T_map,
            contraction_dims).reshape(output_map.dimensions()).eval()};

        forge_eval(output_map, contract);

        if (using_bias) {
            auto bias_map    { bias.as_eigen<scalar_t>() };
            auto bias_bcast_map  {bias_cast.as_eigen<scalar_t>()};
            auto opt_dims {output_map.dimensions()};

            assert(bias_map.dimension(0) == 1 && bias_map.dimension(1) == 1 &&
                bias_map.dimension(2) == 1 && "bias must be (1,1,1,output_size)");

            Eigen::array<Eigen::Index , 4> broadcast_dims {opt_dims[0], opt_dims[1], opt_dims[2], 1};

            forge_eval (bias_bcast_map, output_map+bias_map.broadcast(broadcast_dims).eval());
            forge_eval(output_map, bias_bcast_map);
        }
    });
}
