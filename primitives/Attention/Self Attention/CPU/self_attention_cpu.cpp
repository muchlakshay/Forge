#include "self_attention_cpu.h"
#include "Activations/activations.h"
#include "Attention/Self Attention/self_attention_grads_abstract.h"
#include "autograd/attach_node.h"
#include "Linear/LinearLayer.h"

template<typename T>
concept isSubscriptable = requires (T a) {a[0];a.size();};

void Forge::SelfAttentionCPU::forward(const Tensor& input, const Tensor& query_W, const Tensor& key_W, const Tensor& value_W,
    Tensor& output, const Tensor& mask, const Linear& linear, std::size_t heads, std::size_t d_model, bool using_mask) const {
    Softmax softmax;
    bool need_grads {input.need_grads() || query_W.need_grads() || key_W.need_grads() || value_W.need_grads()};
    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {

        auto inp_map {input.as_eigen<scalar_t, 3>()};
        auto Q_W_map {query_W.as_eigen<scalar_t, 3>()};
        auto K_W_map {key_W.as_eigen<scalar_t, 3>()};
        auto V_W_map {value_W.as_eigen<scalar_t, 3>()};

        auto batch_size {static_cast<std::size_t>(inp_map.dimensions()[0])};
        auto seq_len {static_cast<std::size_t>(inp_map.dimensions()[1])};
        auto V_dims {static_cast<std::size_t>(V_W_map.dimensions()[2])};
        auto Q_K_dims {query_W.shape().back()};

        auto dtype {input.dtype()};

        Eigen::array contract_dims_1 {Eigen::IndexPair<std::size_t>(2, 1)};
        auto Q {inp_map.contract(Q_W_map, contract_dims_1)};
        auto K {inp_map.contract(K_W_map, contract_dims_1)};
        auto V {inp_map.contract(V_W_map, contract_dims_1)};

        Eigen::array<std::size_t, 4> shuffling_dims {0, 2, 1, 3};
        auto Q_T {Q.shuffle(shuffling_dims)};
        auto K_T {K.shuffle(shuffling_dims)};
        auto V_T {V.shuffle(shuffling_dims)};

        Tensor Q_dot_K {{batch_size, heads, seq_len, seq_len}, dtype, need_grads};
        auto Q_dot_K_map {Q_dot_K.as_eigen<scalar_t>()};
        Eigen::array contract_dims_2 {Eigen::IndexPair<std::size_t>(1, 1)};

        for (std::size_t B {}; B<batch_size; ++B) {
            for (std::size_t H {}; H<heads; ++H) {
                auto Q_slice {Q_T.chip(B, 0).chip(H, 0)};
                auto K_slice {K_T.chip(B, 0).chip(H, 0)};

                auto atten_scores {Q_slice.contract(K_slice, contract_dims_2)};

                auto Q_dot_K_map_slice {Q_dot_K_map.chip(B, 0).chip(H, 0)};
                forge_eval(Q_dot_K_map_slice, atten_scores);
            }
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
        auto atten_dot_V_map {atten_dot_V.as_eigen<scalar_t>()};
        Eigen::array contract_dims_3 {Eigen::IndexPair<std::size_t>(1, 0)};


        for (std::size_t B {}; B<batch_size; ++B) {
            for (std::size_t H {}; H<heads; ++H) {
                auto atten_slice {atten_scores_map.chip(B, 0).chip(H, 0)};
                auto V_slice {V_T.chip(B, 0).chip(H, 0)};

                auto result {atten_slice.contract(V_slice, contract_dims_3)};
                auto atten_dot_V_map_slice {atten_dot_V_map.chip(B, 0).chip(H, 0)};
                forge_eval(atten_dot_V_map_slice, result);
            }
        }
        Tensor atten_dot_V_reshaped {{batch_size, seq_len, heads * V_dims}, dtype, need_grads};
        auto atten_dot_V_reshaped_map {atten_dot_V_reshaped.as_eigen<scalar_t>()};
        Eigen::array<std::size_t, 4> reshape_dims {1, batch_size, seq_len, heads * V_dims};
        auto AdVr {atten_dot_V_map.shuffle(shuffling_dims).reshape(reshape_dims)};
        forge_eval(atten_dot_V_reshaped_map, AdVr);


        auto opt_l {linear(atten_dot_V_reshaped)};
        output = opt_l.reshape(batch_size, seq_len, d_model);
        output.gradients() = output.gradients().reshape(batch_size, seq_len, d_model);

        if (need_grads) {
           attach_node<SelfAttentionGradsAbstract, 9>(output, input.device(), primitive_dispatcher(),
            primitive_ops::selfAttention, input, query_W, key_W, value_W, Q_dot_K, atten_scores, atten_dot_V_reshaped, mask, opt_l);
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

