#include "../../include/Forge.h"
using namespace Forge;

int sample_top_k(Tensor& probs, int k) {
    auto* ptr {static_cast<float*>(probs.data())};
    int vocab_size = probs.size();
    std::vector<std::pair<int, float>> idx_probs;
    for (int i{}; i<vocab_size; i++) idx_probs.emplace_back(i, ptr[i]);
    std::sort(idx_probs.begin(), idx_probs.end(), [](const auto& a, const auto& b){return a.second > b.second;});

    std::vector top_k_pool (idx_probs.begin(), idx_probs.begin()+k);
    float sum {0.0f};
    for (const auto& item : top_k_pool) sum += item.second;
    for (auto& item : top_k_pool) item.second /= sum;

    std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution dis(0.0f, 1.0f);
    float random_value {dis(gen)};

    float cumulative_sum {0.0f};
    for (const auto& item : top_k_pool) {
        cumulative_sum += item.second;
        if (random_value <= cumulative_sum) return item.first;
    }

    return top_k_pool[0].first;

}

auto argmax(const Tensor& x) {
    auto* ptr {static_cast<float*>(x.data())};
    auto max_it {std::max_element(ptr, ptr+x.size())};
    auto argmax {std::distance(ptr, max_it)};
    return argmax;
}

auto encode(const std::string& str, const SimpleTokenizer& tk) {
    return tk.encode(str, [](std::string& c) {
        std::string replace_with {"Ġ"};
        if (c[0]==' ') c=c.replace(0, 1, replace_with);
    });
}

void transpose(Tensor& A, Tensor& B, const std::vector<int>& perm) {
    if (A.shape().size() != B.shape().size()) throw std::invalid_argument("Different shapes");
    forge_transpose_AVX2<float>(static_cast<float*>(A.data()),
        static_cast<float*>(B.data()), A.strides(), A.shape(), perm, A.size());
}

static constexpr std::size_t D_MODEL{768}, QK_DIMS{64}, V_DIMS{64};
static constexpr std::size_t HEADS{12}, VOCAB_SIZE{50257}, MAX_SEQ_LEN{1024};

struct TransformerBlock {
    LayerNorm m_ln1{D_MODEL, false};
    SelfAttention m_attn1{D_MODEL, QK_DIMS, V_DIMS, HEADS, true, false, true, true};
    LayerNorm m_ln2{D_MODEL, false};
    Linear m_l1{D_MODEL, 3072, false};
    Gelu m_gelu{};
    Linear m_l2{3072, D_MODEL, false};

    Tensor operator()(Tensor& x) {
        auto ln1 {m_ln1(x)};
        auto attn1 {m_attn1(ln1)};
        // std::cout<<"\n"<<"attn:\n"<<attn1<<"\n";
        auto x2 {x.BroadcastAdd(attn1)};

        auto ln2 {m_ln2(x2)};
        auto l1 {m_l1(ln2)};
        auto gelu {m_gelu(l1)};
        auto l2 {m_l2(gelu)};
        auto opt {l2.BroadcastAdd(x2)};
        return opt;
    }
};

struct PosEnc {Tensor m_pe;};
struct Members {
    Embedding m_embed{D_MODEL, VOCAB_SIZE, Dtype::float32, false};
    PosEnc m_pe;
    std::array<TransformerBlock, 12> m_blocks{};
    LayerNorm m_ln{D_MODEL, false};
    Linear m_l_opt{1, 1, false, false};
};

struct GPT2 {
    Members m_mem;
    Tensor operator()(Tensor& x) {
        auto seq_len {x.size()};
        auto PE {Tensor::FromHostPtr(static_cast<float*>(m_mem.m_pe.m_pe.data()), {seq_len, D_MODEL}, false)};

        if (auto& mask {m_mem.m_blocks[0].m_attn1.mask()}; mask.size()==0 || mask.shape().back()!=seq_len ) {
            for (auto& block:m_mem.m_blocks) block.m_attn1.createMask(seq_len);
        }

        auto embd {m_mem.m_embed(x)};
        auto pe_added {embd.BroadcastAdd(PE)};
        auto block_opt {pe_added};
        for (auto& block:m_mem.m_blocks) block_opt = block(block_opt);
        // std::cout<<"\nMask: "<<m_mem.m_blocks[0].m_attn1.mask()<<"\n";
        auto ln {m_mem.m_ln(block_opt)};
        auto opt {m_mem.m_l_opt(ln)};
        return opt;
    }
};

void load_gpt2(Members& mem, const std::string& filename) {
    auto gpt_state_dict {load_safetensors(filename)};

    mem.m_embed.all_embeddings() = gpt_state_dict["wte.weight"];
    mem.m_pe.m_pe = gpt_state_dict["wpe.weight"];
    mem.m_ln.gamma() = gpt_state_dict["ln_f.weight"];
    mem.m_ln.beta() = gpt_state_dict["ln_f.bias"];
    mem.m_l_opt.weights() = gpt_state_dict["wte.weight"];
    mem.m_l_opt.set_shape(D_MODEL, VOCAB_SIZE);

    for (int i {}; i<12; ++i) {
        auto& block {mem.m_blocks[i]};
        block.m_ln1.gamma() = gpt_state_dict[std::format("h.{}.ln_1.weight", i)];
        block.m_ln1.beta() = gpt_state_dict[std::format("h.{}.ln_1.bias", i)];
        block.m_ln2.gamma() = gpt_state_dict[std::format("h.{}.ln_2.weight", i)];
        block.m_ln2.beta() = gpt_state_dict[std::format("h.{}.ln_2.bias", i)];

        transpose(gpt_state_dict[std::format("h.{}.mlp.c_fc.weight", i)], block.m_l1.weights(), {1, 0});
        block.m_l1.bias() = gpt_state_dict[std::format("h.{}.mlp.c_fc.bias", i)];

        transpose(gpt_state_dict[std::format("h.{}.mlp.c_proj.weight", i)], block.m_l2.weights(), {1, 0});
        block.m_l2.bias() = gpt_state_dict[std::format("h.{}.mlp.c_proj.bias", i)];

        auto& attn_w {gpt_state_dict[std::format("h.{}.attn.c_attn.weight", i)]};
        // std::cout<<attn_w<<"\n";
        Tensor qkv_t {{D_MODEL*3, D_MODEL}, Dtype::float32, false};
        transpose(attn_w, qkv_t, {1, 0});

        std::size_t component_size {D_MODEL*D_MODEL};

        auto* ptr {static_cast<float*>(qkv_t.data())};
        auto qw {Tensor::FromHostPtr(ptr, {12, 64, 768}, false).clone()};
        auto kw {Tensor::FromHostPtr(ptr+component_size, {12, 64, 768}, false).clone()};
        auto vw {Tensor::FromHostPtr(ptr+2*component_size, {12, 64, 768}, false).clone()};

        transpose(qw,block.m_attn1.query(), {0, 2, 1});
        transpose(kw,block.m_attn1.key(), {0, 2, 1});
        transpose(vw,block.m_attn1.value(), {0, 2, 1});

        auto* bptr {static_cast<float*>(gpt_state_dict[std::format("h.{}.attn.c_attn.bias", i)].data())};
        block.m_attn1.query_bias() = Tensor::FromHostPtr(bptr, {HEADS, QK_DIMS}, false).clone();
        block.m_attn1.key_bias() = Tensor::FromHostPtr(bptr+D_MODEL, {HEADS, QK_DIMS}, false).clone();
        block.m_attn1.value_bias() = Tensor::FromHostPtr(bptr+2*D_MODEL, {HEADS, QK_DIMS}, false).clone();

        transpose(gpt_state_dict[std::format("h.{}.attn.c_proj.weight", i)], block.m_attn1.linear().weights(), {1, 0});
        block.m_attn1.linear().bias() = gpt_state_dict[std::format("h.{}.attn.c_proj.bias", i)];
    }

}

std::string clean_token(std::string tok) {
    size_t pos;
    while ((pos = tok.find("Ġ")) != std::string::npos) tok.replace(pos, 2, " ");
    while ((pos = tok.find("Ċ")) != std::string::npos) tok.replace(pos, 2, "\n");
    while ((pos = tok.find("ĉ")) != std::string::npos) tok.replace(pos, 2, "\t");
    while ((pos = tok.find("Âł")) != std::string::npos) tok.replace(pos, 4, "\u00a0");
    return tok;
}

int main() {
    return 0;
}