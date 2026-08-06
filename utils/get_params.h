#pragma once
#include "tensor.h"
#include <rfl.hpp>
#include <type_traits>

namespace Forge {
    template <typename T>
    concept HasParameters = requires(T x){
        {x.parameters()}->std::same_as<std::vector<Tensor*>>;
    };
    template <typename T>
    std::vector<Tensor*> extract_parameters(T& model);
}

template<typename T>
std::vector<Forge::Tensor *> Forge::extract_parameters(T& model) {
    auto data_members {rfl::to_view(model)};
    std::vector<Tensor*> parameters;

    rfl::apply(
        [&](auto*... members) {
            ([&]() {
                if constexpr (HasParameters<std::remove_pointer_t<decltype(members)>>) {
                    for (std::vector<Tensor*> params {members->parameters()}; auto p : params) {parameters.push_back(p);}
                }
            }(), ...);
        }, data_members.values());
    return parameters;
}
