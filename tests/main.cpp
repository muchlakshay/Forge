#include "execution_ctx.h"
#include "../Forge.h"
#include "../data/preprocessing/tokenizer.h"
#include "ops/Maths/cpu_blas.h"
#include <chrono>

std::string read_file(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) throw std::runtime_error("File could not be opened");
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

auto get_token_ids(const std::string& file_name, SimpleTokenizer& tk){
    static bool loaded {};
    if (!loaded) {
        tk.load(R"(C:\Users\MSI\CLionProjects\Forge\tests\saved_2.tok)");
        // tk.on_file(R"(C:\Users\MSI\CLionProjects\Forge\tests\text.txt)", 5000);
        // tk.save(R"(C:\Users\MSI\CLionProjects\Forge\tests\saved_2.tok)");
        loaded = true;
    }

    auto train_text {read_file(file_name)};
    auto tok_ids_eigen {tk.encode(train_text)};
    std::vector<int> token_ids_vec {tok_ids_eigen.data(), tok_ids_eigen.data()+tok_ids_eigen.size()};
    return token_ids_vec;
}

auto get_prediction_tokens(Forge::Tensor& predictions, SimpleTokenizer& tk) {
    std::string result_string {};
    auto pred_map {predictions.as_eigen<float, 4>()};
    auto seq_len {pred_map.dimensions()[2]};

    std::vector<Eigen::Index> argmax_indices(seq_len);

    Eigen::TensorMap<Eigen::Tensor<Eigen::Index, 1, Eigen::RowMajor>> argmax_map(argmax_indices.data(), seq_len);

    argmax_map = pred_map.argmax(3).reshape(Eigen::array{static_cast<Eigen::Index>(seq_len)});

    auto token_ids {tk.getIds()};
    for (std::size_t i = 0; i < seq_len; ++i) {
        int predicted_token_id = static_cast<int>(argmax_map(i));

        for (auto& m : token_ids) {
            if (m.second == predicted_token_id) {
                result_string += m.first;
                break;
            }
        }
    }
    return result_string;
}


auto get_prediction_token(Forge::Tensor& pred, SimpleTokenizer& tk) {
    auto last_row {static_cast<float*>(pred.data())+ (pred.shape()[0]-1)*pred.shape()[1]};
    int argmax {};
    float val {*last_row};
    for (std::size_t i {}; i<pred.shape()[1]; ++i) {
        if (last_row[i]>val) {
            argmax = static_cast<int>(i);
            val = last_row[i];
        }
    }
    for (auto& m : tk.getIds()) {
        if (m.second==argmax) return m.first;
    }
}

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
