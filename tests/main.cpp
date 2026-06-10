#include "execution_ctx.h"
#include "../Forge.h"
#include "../data/preprocessing/tokenizer.h"
#include "ops/Maths/cpu_blas.h"
#include <chrono>


struct Model {
    Forge::Embedding m_embed;
    Forge::SelfAttention m_atten_1;
    Forge::SelfAttention m_atten_2;
    Forge::Linear m_linear_1;
    Forge::Linear m_linear_2;
    Forge::Linear m_linear_3;
    Forge::Linear m_linear_4;
    Forge::Linear m_linear_5;
    Forge::LayerNorm m_layerNorm_1;
    Forge::LayerNorm m_layerNorm_2;
    Forge::LayerNorm m_layerNorm_3;
    Forge::LayerNorm m_layerNorm_4;
    Forge::LayerNorm m_layerNorm_5;
    Forge::Tensor m_PE {};
    Forge::Gelu m_gelu {};
    Forge::Softmax m_softmax {};

    Model(std::size_t seq_len, std::size_t d_model, std::size_t vocab_size, std::size_t QK_dims,
        std::size_t V_dims, std::size_t heads) : m_embed {d_model, vocab_size},
          m_atten_1{d_model, QK_dims, V_dims, heads, true, Forge::Device::CPU},
          m_atten_2{d_model, QK_dims, V_dims, heads, true, Forge::Device::CPU},
          m_linear_1{d_model, 256}, m_linear_2{256, d_model}, m_linear_3{d_model, 256}, m_linear_4{256, d_model},
          m_linear_5{d_model, vocab_size},
          m_layerNorm_1{d_model, Forge::Dtype::float32, Forge::Device::CPU},
          m_layerNorm_2{d_model, Forge::Dtype::float32, Forge::Device::CPU},
          m_layerNorm_3{d_model, Forge::Dtype::float32, Forge::Device::CPU},
          m_layerNorm_4{d_model, Forge::Dtype::float32, Forge::Device::CPU},
          m_layerNorm_5{d_model, Forge::Dtype::float32, Forge::Device::CPU},
          m_PE{Forge::sinusoidal_pe(seq_len, d_model)} {
          m_atten_1.createMask(seq_len);
          m_atten_2.createMask(seq_len);
    }

    auto operator()(Forge::Tensor& seq_ids) {
        auto embeddings {m_embed(seq_ids)};
        auto pe_added {embeddings.BroadcastAdd(m_PE)};

        auto seq_len {seq_ids.size()};
        auto d_model {m_embed.d_model()};


        auto LN_1 {m_layerNorm_1(pe_added)};
        auto ctx_embd_1 {m_atten_1(LN_1).reshape(seq_len, d_model)};
        auto residual_1 {ctx_embd_1.BroadcastAdd(pe_added)};

        auto LN_2 {m_layerNorm_2(residual_1)};
        auto FF_1 {m_linear_2(m_gelu(m_linear_1(LN_2)))};
        auto block1_opt {FF_1.BroadcastAdd(residual_1)};

        auto LN_3 {m_layerNorm_3(block1_opt)};

        auto ctx_embd_2 {m_atten_2(LN_3).reshape(seq_len, d_model)};
        auto residual_2 {ctx_embd_2.BroadcastAdd(block1_opt)};

        auto LN_4 {m_layerNorm_4(residual_2)};
        auto FF_2 {m_linear_4(m_gelu(m_linear_3(LN_4)))};
        auto block2_opt {FF_2.BroadcastAdd(residual_2)};

        auto LN_5 {m_layerNorm_5(block2_opt)};
        auto linear_5_opt {m_linear_5(LN_5)};

        return linear_5_opt;
    }

    auto parameters() {
        std::vector params {
            Forge::Parameter{&m_embed.all_embeddings(), true},
            Forge::Parameter{&m_atten_1.query(), true},
            Forge::Parameter{&m_atten_1.key(), true},
            Forge::Parameter{&m_atten_1.value(), true},
            Forge::Parameter{&m_atten_2.query(), true},
            Forge::Parameter{&m_atten_2.key(), true},
            Forge::Parameter{&m_atten_2.value(), true},
            Forge::Parameter{&m_linear_1.weights(), true},
            Forge::Parameter{&m_linear_1.bias(), false},
            Forge::Parameter{&m_linear_2.weights(), true},
            Forge::Parameter{&m_linear_2.bias(), false},
            Forge::Parameter{&m_linear_3.weights(), true},
            Forge::Parameter{&m_linear_3.bias(), false},
            Forge::Parameter{&m_linear_4.weights(), true},
            Forge::Parameter{&m_linear_4.bias(), false},
            Forge::Parameter{&m_linear_5.weights(), true},
            Forge::Parameter{&m_linear_5.bias(), false},
            Forge::Parameter{&m_layerNorm_1.gamma(), false},
            Forge::Parameter{&m_layerNorm_1.beta(), false},
            Forge::Parameter{&m_layerNorm_2.gamma(), false},
            Forge::Parameter{&m_layerNorm_2.beta(), false},
            Forge::Parameter{&m_layerNorm_3.gamma(), false},
            Forge::Parameter{&m_layerNorm_3.beta(), false},
            Forge::Parameter{&m_layerNorm_4.gamma(), false},
            Forge::Parameter{&m_layerNorm_4.beta(), false},
            Forge::Parameter{&m_layerNorm_5.gamma(), false},
            Forge::Parameter{&m_layerNorm_5.beta(), false},
        };
        return params;
    }
    auto& embed(){return m_embed;}
};

int main() {

    return 0;
}
