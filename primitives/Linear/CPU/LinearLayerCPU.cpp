#include "LinearLayerCPU.h"
#include "../../../core/ops/Maths/CPU/math_cpu.h"

#define EigenMatrixMap(dtype, data, rows, cols) \
    Eigen::Map<Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, rows, cols)

void Forge::LinearCPU::forward(const Tensor& input, const Tensor& output, const Tensor& weights,
    const Tensor& bias, bool using_bias) const {

    auto& idims {input.shape()};
    auto& odims {output.shape()};

    std::size_t M {1};
    std::size_t N {weights.shape()[0]};
    std::size_t K {weights.shape()[1]};

    for (int i {}; i<idims.size()-1; ++i) M*=idims[i];

    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        forge_contract<scalar_t>(static_cast<scalar_t*>(input.data()), static_cast<scalar_t*>(weights.data()),
            static_cast<scalar_t*>(output.data()), M, N, K, false, true, true);

        // auto weights_map { weights.as_eigen<scalar_t>()};
        // auto input_map   { input.as_eigen<scalar_t>()  };
        //auto output_map  { output.as_eigen<scalar_t>() };
        // Eigen::array<std::size_t , 4> shuffling {0, 1, 3, 2};
        // Eigen::array<Eigen::IndexPair<std::size_t>, 1> contraction_dims {Eigen::IndexPair<std::size_t>(3, 2)};
        // auto weights_map { weights.as_eigen<scalar_t>()};
        // auto input_map   { input.as_eigen<scalar_t>()  };
        // auto output_map  { output.as_eigen<scalar_t>() };

        // forge_eval(output_map, input_map.contract(weights_map.shuffle(shuffling),
        //     contraction_dims).reshape(output_map.dimensions()));

        if (using_bias) {
            // auto weights_map { weights.as_eigen<scalar_t>()};
            // auto input_map   { input.as_eigen<scalar_t>()  };
            auto output_map  { output.as_eigen<scalar_t>() };
            auto bias_map    { bias.as_eigen<scalar_t>() };
            auto odims {output_map.dimensions()};

            // assert(bias_map.dimension(0) == 1 && bias_map.dimension(1) == 1 &&
            //     bias_map.dimension(2) == 1 && "bias must be (1,1,1,output_size)");
            //
            // Eigen::array<Eigen::Index , 4> broadcast_dims {odims[0], odims[1], odims[2], 1};
            //
            // forge_eval(output_map, output_map+bias_map.broadcast(broadcast_dims));

            broadcastADD_AVX2(output_map.data(), bias_map.data(), output_map.data(), (std::size_t)odims[0]*odims[1]*odims[2],
                (std::size_t)odims[3]);
        }
    });
}
