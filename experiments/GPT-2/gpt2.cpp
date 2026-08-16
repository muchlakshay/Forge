#include "../../include/Forge.h"
#include "ops/Maths/CPU/transpose.h"
#include <windows.h>
using namespace Forge;
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

int main() {
    return 0;
}