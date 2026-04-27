#include "self_attention_cpu.h"
#include "Activations/activations.h"
#include "Linear/LinearLayer.h"

template<typename T>
concept isSubscriptable = requires (T a) {a[0];a.size();};

template<isSubscriptable T>
auto dimsAfterContract(T A, T B, std::size_t d_A, std::size_t d_B) {
    const std::size_t num_dims {A.size()+B.size()};
    std::vector<std::size_t> dims;
    dims.reserve(num_dims - 2);
    for (std::size_t i {}; i < A.size(); ++i) {if (i!=d_A) dims.push_back(A[i]);}
    for (std::size_t i {}; i < B.size(); ++i) {if (i!=d_B) dims.push_back(B[i]);}
    return dims;
}

void Forge::SelfAttentionCPU::forward(const Tensor input, const Tensor query_W, const Tensor key_W, const Tensor value_W,
    Tensor output, const Tensor mask, const Linear& linear, std::size_t heads, bool using_mask) const {
    Softmax softmax;

    DISPATCH_ALL_TYPES(input.dtype(), Device::CPU, [&] {
        using TensorRef = Eigen::TensorRef<Eigen::Tensor<scalar_t, 4>>;

        auto inp_map {input.as_eigen<scalar_t>()};
        auto Q_W_map {query_W.as_eigen<scalar_t>()};
        auto K_W_map {key_W.as_eigen<scalar_t>()};
        auto V_W_map {value_W.as_eigen<scalar_t>()};

        Eigen::array contract_dims_1 {Eigen::IndexPair<std::size_t>(3, 2)};
        Eigen::array<Eigen::Index, 4> shuffling_dims {0, 2, 1, 3};
        auto Q {inp_map.contract(Q_W_map, contract_dims_1)};
        auto K {inp_map.contract(K_W_map, contract_dims_1)};
        auto V {inp_map.contract(V_W_map, contract_dims_1)};

        auto Q_T {Q.shuffle(shuffling_dims)};
        auto K_T {K.shuffle(shuffling_dims)};
        auto V_T {V.shuffle(shuffling_dims)};

        auto Q_K_dims {dimsAfterContract(inp_map.dimensions(), Q_W_map.dimensions(), 3, 2)};
        Tensor temp {{dimsAfterContract(Q_K_dims, Q_K_dims, 3, 3)}};
        auto temp_map {temp.as_eigen<scalar_t>()};
        Eigen::array contract_dims_2 {Eigen::IndexPair<std::size_t>(3, 3)};

        temp_map = Q_T.contract(K_T, contract_dims_2) / static_cast<scalar_t>(std::sqrt(query_W.shape().back()));

        if (using_mask) {
            auto mask_map {mask.as_eigen<scalar_t>()};
            auto dims {temp_map.dimensions()};
            Eigen::array bcast_dims {dims[0], dims[1], 1, 1};
            temp_map = temp_map.broadcast(bcast_dims);
        }

        auto temp_2 {softmax(temp)};
        auto temp_2_map {temp_2.as_eigen<scalar_t>()};

        Eigen::array contract_dims_3 {Eigen::IndexPair<std::size_t>(3, 2)};
        auto attention  {temp_2_map.contract(V_T, contract_dims_3)};
        TensorRef shuff_atten {attention.shuffle(shuffling_dims)};
        auto d {shuff_atten.dimensions()};
        Eigen::array reshape_dims {1, d[0], d[1], d[2]*d[3]};
        TensorRef concat_expr {shuff_atten.reshape(reshape_dims)};
        auto opt_dims {concat_expr.dimensions()};
        Tensor concatinated {std::vector<std::size_t>(opt_dims.begin(), opt_dims.end()), query_W.dtype()};

        auto concat_map {concatinated.as_eigen<scalar_t>()};
        concat_map = concat_expr;
        output = linear(concatinated);
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

