#include "Attention/Self Attention/self_attention_impl_abstract.h"

namespace Forge {
    struct SelfAttentionCPU;
}

struct Forge::SelfAttentionCPU : SelfAttentionImplAbstract {
    void forward(Tensor input, Tensor query_W, Tensor key_W, Tensor value_W, Tensor output, Tensor mask, Linear& linear,
        std::size_t heads, bool using_mask) const override;
};