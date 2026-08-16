#include "self_attention_cpu.h"
#include "Activations/activations.h"
#include "Attention/Self Attention/self_attention_grads_abstract.h"
#include "autograd/attach_node.h"
#include "Linear/LinearLayer.h"
#include "../../../../core/ops/Maths/CPU/math_cpu.h"

template<typename T>
concept isSubscriptable = requires (T a) {a[0];a.size();};

void Forge::SelfAttentionCPU::forward(const Tensor &input, const Tensor &query_W, const Tensor &key_W, const Tensor &value_W,
                                      Tensor &output, const Tensor &mask, const Tensor& Q_bias, const Tensor& K_bias,
                                      const Tensor &V_bias, const Linear &linear, std::size_t heads, std::size_t d_model,
                                      bool using_mask) const {

    // std::cout<<"\nQW:\n"<<query_W<<"\nKW:\n"<<key_W<<"\nVW:\n"<<value_W<<"\n";
    Softmax softmax;
    bool need_grads {input.need_grads() || query_W.need_grads() || key_W.need_grads() || value_W.need_grads()
        || Q_bias.need_grads() || K_bias.need_grads() || V_bias.need_grads() || linear.bias().need_grads()
        || linear.weights().need_grads()};

    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        auto inp_map {input.as_eigen<scalar_t, 3>()};
        auto Q_W_map {query_W.as_eigen<scalar_t, 3>()};
        auto K_W_map {key_W.as_eigen<scalar_t, 3>()};
        auto V_W_map {value_W.as_eigen<scalar_t, 3>()};

        auto batch_size {static_cast<std::size_t>(inp_map.dimensions()[0])};
        auto seq_len {static_cast<std::size_t>(inp_map.dimensions()[1])};
        auto V_dims {static_cast<std::size_t>(V_W_map.dimensions()[2])};
        auto Q_K_dims {query_W.shape().back()};

        auto Q_tensor {Tensor{{batch_size, heads, seq_len, Q_K_dims}, input.dtype(), false}};
        auto K_tensor {Tensor{{batch_size, heads, seq_len, Q_K_dims}, input.dtype(), false}};
        auto V_tensor {Tensor{{batch_size, heads, seq_len, V_dims}, input.dtype(), false}};
        auto Q_T {Q_tensor.as_eigen<scalar_t>()};
        auto K_T {K_tensor.as_eigen<scalar_t>()};
        auto V_T {V_tensor.as_eigen<scalar_t>()};

        auto dtype {input.dtype()};

        Eigen::array contract_dims_1 {Eigen::IndexPair<std::size_t>(2, 1)};
        auto Q {inp_map.contract(Q_W_map, contract_dims_1)};
        auto K {inp_map.contract(K_W_map, contract_dims_1)};
        auto V {inp_map.contract(V_W_map, contract_dims_1)};

        Eigen::array<std::size_t, 4> shuffling_dims {0, 2, 1, 3};
        forge_eval(Q_T, Q.shuffle(shuffling_dims));
        forge_eval(K_T, K.shuffle(shuffling_dims));
        forge_eval(V_T, V.shuffle(shuffling_dims));

        // std::cout<<"Q_T:\n"<<Q_tensor<<"\n";

        if (Q_bias.size()!=0) {
            Eigen::array<std::size_t, 4> bcast_dims {batch_size, 1, seq_len, 1};
            forge_eval(Q_T, Q_T+Q_bias.as_eigen<scalar_t>().shuffle(shuffling_dims).broadcast(bcast_dims));
            forge_eval(K_T, K_T+K_bias.as_eigen<scalar_t>().shuffle(shuffling_dims).broadcast(bcast_dims));
            forge_eval(V_T, V_T+V_bias.as_eigen<scalar_t>().shuffle(shuffling_dims).broadcast(bcast_dims));
        }


        Tensor Q_dot_K {{batch_size, heads, seq_len, seq_len}, dtype, need_grads};
        auto Q_dot_K_map {Q_dot_K.as_eigen<scalar_t>()};
        Eigen::array contract_dims_2 {Eigen::IndexPair<std::size_t>(1, 1)};

        for (std::size_t batch {}; batch<batch_size * heads; ++batch) {
            forge_contract(
                static_cast<scalar_t*>(Q_tensor.data())+batch*seq_len*Q_K_dims,
                static_cast<scalar_t*>(K_tensor.data())+batch*seq_len*Q_K_dims,
                static_cast<scalar_t*>(Q_dot_K.data())+batch*seq_len*seq_len,
                seq_len, seq_len, Q_K_dims, false, true, true
            );
        }
        forge_eval(Q_dot_K_map, Q_dot_K_map * static_cast<scalar_t>(1)/Eigen::numext::sqrt(static_cast<scalar_t>(Q_K_dims)));

        if (using_mask) {
            auto mask_map {mask.as_eigen<scalar_t>()};
            auto dims = Q_dot_K_map.dimensions();
            Eigen::array<Eigen::Index, 4> bcast_dims {dims[0], dims[1], 1, 1};
            forge_eval(Q_dot_K_map, Q_dot_K_map+ mask_map.broadcast(bcast_dims));
        }
        auto atten_scores {softmax(Q_dot_K)};
        auto atten_scores_map {atten_scores.as_eigen<scalar_t>()};
        Tensor atten_dot_V {{batch_size, heads, seq_len, V_dims}, dtype, need_grads};
        Eigen::array contract_dims_3 {Eigen::IndexPair<std::size_t>(1, 0)};

        auto atten_dot_V_map {atten_dot_V.as_eigen<scalar_t>()};
        for (std::size_t batch {}; batch<batch_size * heads; ++batch) {
            forge_contract(
                static_cast<scalar_t*>(atten_scores.data())+batch*seq_len*seq_len,
                static_cast<scalar_t*>(V_tensor.data())+batch*seq_len*V_dims,
                static_cast<scalar_t*>(atten_dot_V.data())+batch*seq_len*V_dims,
                seq_len, V_dims, seq_len, false, false, true
            );
        }
        Tensor atten_dot_V_reshaped {{batch_size, seq_len, heads, V_dims}, dtype, need_grads};
        auto atten_dot_V_reshaped_map {atten_dot_V_reshaped.as_eigen<scalar_t>()};

        Eigen::array<std::size_t, 4> reshape_dims {batch_size, seq_len, heads * V_dims};

        auto AdVr {atten_dot_V_map.shuffle(shuffling_dims)};
        forge_eval(atten_dot_V_reshaped_map, AdVr);
        atten_dot_V_reshaped = atten_dot_V_reshaped.reshape(batch_size, seq_len, heads * V_dims);

        auto opt_l {linear(atten_dot_V_reshaped)};
        output = opt_l.reshape(batch_size, seq_len, d_model);
        if (output.need_grads()) output.gradients() = output.gradients().reshape(batch_size, seq_len, d_model);
        if (need_grads) {
           attach_node<SelfAttentionGradsAbstract, 12>(output, input.device(), primitive_dispatcher(),
            primitive_ops::selfAttention, input, query_W, key_W, value_W, Q_bias, K_bias, V_bias,
            Q_dot_K, atten_scores, atten_dot_V_reshaped, mask, opt_l);
        }
    });
}

//SHAPES MAP FOR REFERENCE
/*
 * inp (batch, seq_len, d_model), qw/kw = (heads, d_model, q_k_dims), vw = (heads, d_model, V_dims)
 * Q and K = (batch, seq_len, heads, q_k_dims)
 * transpose(0, 2, 1, 3) = (batch, heads, seq_len, q_k_dims)
 * Q@K^T = (batch, heads, seq_len, q_k_dims) @ (batch, heads, q_k_dims, seq_len) = (batch, heads, seq_len, seq_len) = C
 * softmax(C/sqrt(q_k_dims)) = (batch, heads, seq_len, seq_len) = D
 * V = (batch, seq_len, heads, V_dims)
 * transpose(0, 2, 1, 3) = (batch, heads, seq_len, V_dims)
 * D @ V = (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, V_dims) = (batch, heads, seq_len, V_dims) = E
 * transpose (0, 2, 1, 3) = (batch, seq_len, heads, V_dims) = E_
 * Concatenate_Horizontally(E_) = (batch, seq_len, heads*V_dims) = F
 * Linear[W] = (d_model, heads*V_dims)
 * F @ W^T = (batch, seq_len, heads*V_dims) @ (heads*V_dims, d_model) = (batch, seq_len, d_model) = opt
 */

