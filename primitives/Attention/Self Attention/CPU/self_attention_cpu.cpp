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



