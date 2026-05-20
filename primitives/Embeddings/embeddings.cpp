#include "embeddings.h"

#include "embeddings_abstract.h"
#include "embeddings_grads_abstract.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "autograd/attach_node.h"

Forge::Embedding::Embedding(std::size_t d_model, std::size_t vocab_size, Dtype dtype,bool need_grads,
                            Initializers initializer, Device device) : m_embeddings{Tensor::Random({vocab_size, d_model}, dtype, initializer, device, need_grads)},
                                                                       m_d_model{d_model}, m_vocab_size{vocab_size} {}

Forge::Tensor Forge::Embedding::operator()(const Tensor& seq_ids) {
    if (seq_ids.dtype()!=Dtype::int32) throw std::invalid_argument("Seq Ids must be of Dtype int32");
    if (seq_ids.shape().size()!=1) throw std::invalid_argument("Seq Ids must be a 1-D Tensor");

    auto dtype {all_embeddings().dtype()};
    auto device {all_embeddings().device()};

    Tensor opt {{seq_ids.size(), m_d_model}, dtype, m_embeddings.need_grads(), device};
    const auto* impl {primitive_dispatcher().lookup<EmbeddingsImplAbstract>(primitive_ops::Embeddings, opt.dispatch_key())};
    impl->getEmbeddings(seq_ids, m_embeddings, opt);

    return opt;
}
