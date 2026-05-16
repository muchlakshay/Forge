#include "LinearGradsCPU.h"

void Forge::LinearGradsCPU::compute_grads(const Tensor& input, const Tensor& weights, const Tensor& bias,
        const Tensor& output) const{
    using grads_t = float;
    const auto tensor_dtype {input.dtype()};
    Eigen::array<Eigen::IndexPair<std::size_t>, 1> contraction_dims{};

    auto output_grads_map {output.gradients().as_eigen<grads_t>()};
    DISPATCH_ALL_TYPES(tensor_dtype, Device::CPU, [&] {

        if (input.need_grads()) {
            auto input_grads_map {input.gradients().as_eigen<grads_t>()};
            auto weights_map {weights.as_eigen<scalar_t, 2>()};
            contraction_dims[0] = Eigen::IndexPair<std::size_t>(3, 0);
            input_grads_map += output_grads_map.contract(weights_map.template cast<grads_t>(),
            contraction_dims).reshape(input_grads_map.dimensions());
            // std::cout<<"L W:\n"<<output.gradients()<<"\n";
            // opt_grads (batch, seq_len, d_model) @ W (d_model, V_dims * heads) = AcVr (batch, heads, seq_len, V_dims*heads)
        }

        if (weights.need_grads()) {
            auto weights_grads_map {weights.gradients().as_eigen<grads_t>()};
            auto input_map {input.as_eigen<scalar_t>()};

            auto X {output_grads_map.dimension(0)}, Y {output_grads_map.dimension(1)};
            auto batch_size {output_grads_map.dimension(2)}, output_size {output_grads_map.dimension(3)};
            auto input_size {input_map.dimension(3)};

            auto output_grads_map_2D {output_grads_map.reshape(Eigen::array<Eigen::Index, 2>{X*Y*batch_size, output_size})};
            auto input_map_2D {input_map.reshape(Eigen::array<Eigen::Index, 2>{X*Y*batch_size, input_size})};

            Eigen::array dims {Eigen::IndexPair<std::size_t>(1, 0)};

            weights_grads_map += (
                output_grads_map_2D.shuffle(
                    Eigen::array<Eigen::Index, 2>{1, 0}).contract(input_map_2D.template cast<grads_t>(), dims)
                    ).reshape(weights_grads_map.dimensions());
            // std::cout<<"W Grads:" << weights.gradients()<<"\n";
        }
        if (bias.need_grads()) {
            auto bias_grads_map {bias.gradients().as_eigen<grads_t>()};
            Eigen::array<Eigen::Index, 3> reduction_axis {0, 1, 2};
            bias_grads_map+=output_grads_map.sum(reduction_axis).reshape(bias_grads_map.dimensions());
        }
    });
}