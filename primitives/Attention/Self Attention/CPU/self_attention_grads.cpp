#include "self_attention_grads_cpu.h"
#include "tensor.h"

void Forge::SelfAttentionGradsCPU::compute_grads(const Tensor &inp, const Tensor &Q_W, const Tensor &K_W, const Tensor &V_W,
    const Tensor &QcKs, const Tensor &atten_scores, const Tensor &AcVr, const  Tensor& mask, const Tensor& opt_l,
    const Tensor &opt) const {
    using grads_t = float;

    opt_l.gradients() = opt.gradients();
    DISPATCH_ALL_TYPES(inp.dtype(), inp.device(), [&] {
        auto inp_map {inp.as_eigen<scalar_t, 3>()};
        auto Q_W_map {Q_W.as_eigen<scalar_t, 3>()};
        auto K_W_map {K_W.as_eigen<scalar_t, 3>()};
        auto V_W_map {V_W.as_eigen<scalar_t, 3>()};

        Eigen::array<std::size_t, 4> shuffling_dims {0, 2, 1, 3};
        Eigen::array<std::size_t, 4> shuffling_dims_2 {0, 2, 1};
        Eigen::array<std::size_t, 2> shuffling_dims_3 {1, 0};

        Eigen::array contract_dims_1 {Eigen::IndexPair<std::size_t>(2, 1)};

        auto batch_size {static_cast<std::size_t>(inp_map.dimensions()[0])};
        auto seq_len {static_cast<std::size_t>(inp_map.dimensions()[1])};
        auto Q_K_dims {Q_W.shape().back()};
        auto V_dims {static_cast<std::size_t>(V_W_map.dimensions()[2])};
        auto heads {static_cast<std::size_t>(Q_W.shape()[0])};
        auto d_model {static_cast<std::size_t>(inp.shape()[2])};

        Tensor Q_T_t {{batch_size, heads, seq_len, Q_K_dims}, Q_W.dtype(), false};
        Tensor K_T_t {{batch_size, heads, seq_len, Q_K_dims}, K_W.dtype(), false};
        Tensor V_T_t {{batch_size, heads, seq_len, V_dims  }, V_W.dtype(), false};

        auto Q_T {Q_T_t.as_eigen<scalar_t>()};
        auto K_T {K_T_t.as_eigen<scalar_t>()};
        auto V_T {V_T_t.as_eigen<scalar_t>()};

        forge_eval(Q_T, inp_map.contract(Q_W_map, contract_dims_1).shuffle(shuffling_dims));
        forge_eval(K_T, inp_map.contract(K_W_map, contract_dims_1).shuffle(shuffling_dims));
        forge_eval(V_T, inp_map.contract(V_W_map, contract_dims_1).shuffle(shuffling_dims));


        auto AcVr_grads_map {AcVr.gradients().as_eigen<grads_t>()};
        opt_l.backward();


        Eigen::array<std::size_t, 4> reshaping_dims {batch_size, seq_len, heads, V_dims};
        auto reshape_shuff_expr {AcVr_grads_map.reshape(reshaping_dims).shuffle(shuffling_dims)};
        forge_eval(AcVr_grads_map, reshape_shuff_expr);

        auto atten_scores_map {atten_scores.as_eigen<scalar_t>()};
        auto atten_scores_grads_map {atten_scores.gradients().as_eigen<grads_t>()};

        Eigen::array contract_dims_2 {Eigen::IndexPair<std::size_t>(1, 0)};

        Eigen::array contract_dims_3 {Eigen::IndexPair<std::size_t>(0, 0), Eigen::IndexPair<std::size_t>(1, 1)};
        Eigen::array contract_dims_4 {Eigen::IndexPair<std::size_t>(2, 0), Eigen::IndexPair<std::size_t>(3, 1)};
        Eigen::array contract_dims_6 {Eigen::IndexPair<std::size_t>(3, 1), Eigen::IndexPair<std::size_t>(2, 0)};
        Eigen::array contract_dims_7 {Eigen::IndexPair<std::size_t>(0, 0), Eigen::IndexPair<std::size_t>(2, 2)};

        Eigen::array<std::size_t, 3> shuff_dims {1, 0, 2}, shuff_dims_2 {0, 2, 1};



        auto V_grads_T {Tensor::Zeros({batch_size, heads, seq_len, V_dims}, false, Dtype::float32)};
        auto V_grads_T_map {V_grads_T.as_eigen<grads_t>()};

        Tensor As_shuff {{seq_len, seq_len}, Dtype::float32, false};
        auto As_shuff_map {As_shuff.as_eigen<grads_t, 2>()};

        for (std::size_t B{}; B<batch_size; ++B) {
            for (std::size_t H{}; H<heads; ++H) {

                auto AcV_grads_slice {AcVr_grads_map.chip(B, 0).chip(H, 0)};
                auto atten_scores_slice {atten_scores_map.chip(B, 0).chip(H, 0)};

                forge_eval(As_shuff_map, atten_scores_slice.template cast<grads_t>().shuffle(
                shuffling_dims_3));

                auto V_grads_slice {As_shuff_map.contract(AcV_grads_slice, contract_dims_2)};

                auto V_grads_T_slice_map {V_grads_T_map.chip(B, 0).chip(H, 0)};
                forge_eval(V_grads_T_slice_map, V_grads_slice);

                auto V_T_slice {V_T.chip(B, 0).chip(H, 0).template cast<grads_t>().eval()};
                auto As_grads_slice {atten_scores_grads_map.chip(B, 0).chip(H, 0)};
                forge_eval(As_grads_slice, AcV_grads_slice.contract(V_T_slice, contract_dims_2));
            }
        }

        // std::cout<<"\n"<<"V grads: "<<V_grads_T<<"\n";
        auto V_grads_map {V_grads_T_map.shuffle(shuffling_dims)};
        if (V_W.need_grads()) {
            Tensor temp {inp.shape(), Dtype::float32, false};
            auto temp_map {temp.as_eigen<grads_t, 3>()};

            auto V_W_grads_map {V_W.gradients().as_eigen<grads_t, 3>()};
                forge_eval(temp_map, inp_map.template cast<grads_t>());

            forge_eval(V_W_grads_map, V_W_grads_map+temp_map.contract(
             V_grads_map, contract_dims_3).shuffle(shuff_dims));
        }

        if (inp.need_grads()) {
            Tensor temp {{heads, V_dims, seq_len}, Dtype::float32, false};
            auto temp_map {temp.as_eigen<grads_t, 3>()};
            auto inp_grads_map {inp.gradients().as_eigen<grads_t, 3>()};

            forge_eval(temp_map, V_W_map.template cast<grads_t>().shuffle(shuff_dims_2));

            forge_eval(inp_grads_map, inp_grads_map + V_grads_map.contract(
            temp_map, contract_dims_4));
        };

        auto QcKs_grads_map {QcKs.gradients().as_eigen<grads_t>()};
        atten_scores.backward(false, true);

        auto QcK_grads {QcKs_grads_map*QcKs_grads_map.constant(
            static_cast<grads_t>(1)/Eigen::numext::sqrt(static_cast<grads_t>(Q_K_dims)))};

        auto Q_K_grads {Tensor::Zeros({batch_size, heads, seq_len, Q_K_dims}, false, Dtype::float32)};
        auto Q_K_grads_map {Q_K_grads.as_eigen<grads_t>()};

        Tensor QcK_grads_slice {{seq_len, seq_len}, Dtype::float32, false};
        auto QcK_grads_slice_map {QcK_grads_slice.as_eigen<grads_t, 2>()};

        for (std::size_t B{}; B<batch_size; ++B) {
            for (std::size_t H{}; H<heads; ++H) {

                forge_eval(QcK_grads_slice_map, QcK_grads.chip(B, 0).chip(H, 0));
                auto K_T_slice{K_T.chip(B, 0).chip(H, 0).template cast<grads_t>().eval()};

                auto Q_K_grads_slice_map {Q_K_grads_map.chip(B, 0).chip(H, 0)};
                forge_eval(Q_K_grads_slice_map,  QcK_grads_slice_map.contract(K_T_slice, contract_dims_2));
            }
        }
        Tensor inp_shuff {{inp.shape()}, Dtype::float32, false};
        auto inp_shuff_map {inp_shuff.as_eigen<grads_t, 3>()};
        forge_eval(inp_shuff_map, inp_map.shuffle(shuffling_dims_2).template cast<grads_t>());

        Tensor Q_K_W_temp {{heads, d_model, Q_K_dims}, Dtype::float32, false};
        auto Q_K_W_temp_map {Q_K_W_temp.as_eigen<grads_t, 3>()};

         if (inp.need_grads()) {
             auto inp_grads_map {inp.gradients().as_eigen<grads_t, 3>()};

             forge_eval(Q_K_W_temp_map, Q_W_map.shuffle(shuffling_dims_2).template cast<grads_t>());
             forge_eval(inp_grads_map, inp_grads_map + Q_K_grads_map.shuffle(shuffling_dims).contract(
             Q_K_W_temp_map, contract_dims_6));
         }
         if (Q_W.need_grads()) {
             auto Q_W_grads_map {Q_W.gradients().as_eigen<grads_t, 3>()};

             forge_eval(Q_W_grads_map, Q_W_grads_map + inp_shuff_map.contract (
             Q_K_grads_map, contract_dims_7));
         }

        Q_K_grads_map.setConstant(0.f);


         for (std::size_t B{}; B<batch_size; ++B) {
             for (std::size_t H{}; H<heads; ++H) {
                 forge_eval(QcK_grads_slice_map, QcK_grads.chip(B, 0).chip(H, 0).shuffle(shuffling_dims_3));

                 auto Q_T_slice{Q_T.chip(B, 0).chip(H, 0).template cast<grads_t>()};
                 auto Q_K_grads_slice_map {Q_K_grads_map.chip(B, 0).chip(H, 0)};

                 forge_eval(Q_K_grads_slice_map, QcK_grads_slice_map.contract(
                     Q_T_slice, contract_dims_2));
             }
         }

         if (inp.need_grads()) {
             Tensor temp {{heads, d_model, Q_K_dims}, Dtype::float32, false};

             // auto temp_map {temp.as_eigen<grads_t, 3>()};
             auto inp_grads_map {inp.gradients().as_eigen<grads_t, 3>()};

             forge_eval(Q_K_W_temp_map, K_W_map.shuffle(shuffling_dims_2).template cast<grads_t>());
             forge_eval(inp_grads_map, inp_grads_map+Q_K_grads_map.shuffle(shuffling_dims).contract(
             Q_K_W_temp_map, contract_dims_6));
         }

         if (K_W.need_grads()) {
             auto K_W_grads { K_W.gradients().as_eigen<grads_t, 3>()};

             forge_eval(K_W_grads, K_W_grads + inp_shuff_map.contract (
                Q_K_grads_map, contract_dims_7));
            }
    });
}

//SHAPES FOR MAP REFERENCE
/*
 * AcVr_grads = (1, seq_len, heads*V_dims) [after opt.backward()]
 * AcV_grads = (batch_size, heads, seq_len, V_dims) [after reshape and shuffling]
 *
 * V_s = (batch, heads, seq_len, V_dims)
 * atten_scores = (batch, heads, seq_len, seq_len)
 * V_s_grads = atten_scores.T @ AcV_grads = (seq_len, seq_len).T @ (seq_len, V_dims) = (batch, heads, seq_len, V_dims)
 * V_grads = V_s_grads.shuffle(0, 2, 1, 3)
 * atten_scores_grads = (seq_len, V_dims) @ (seq_len, V_dims).T = (seq_len, V_dims) @ (V_dims, seq_len) = (batch, heads, seq_len, seq_len)
 * inp (batch, seq_len, d_model) @ V_grads (batch, seq_len, heads, V_dims) = V_W_grads (heads, d_model, V_dims)
 * V_grads (batch, seq_len, heads, V_dims) @ V_W_T (heads, V_dims, d_model) = inp (batch, seq_len, d_model)
 * Q_grads (batch, heads, seq_len, Q_K_dims) = QcK_grads (batch, heads, seq_len, seq_len) @ K_T (batch, heads, seq_len, Q_K_dims)
 * Q_W_grads (heads, d_model, q_k_dims) = inp (batch, seq_len, d_model).T @ Q_K_grads (batch, heads, seq_len, q_k_dims)
 * K_grads (batch, heads, seq_len, Q_K_dims) = QcK_grads (batch, heads, seq_len, seq_len).T @ Q_T (batch, heads, seq_len, Q_K_dims).T
 * K_W_grads or Q_W_grads (heads, d_model, q_k_dims) = inp (batch, seq_len, d_model).T @ Q_K_grads (batch, heads, seq_len, q_k_dims)
*/