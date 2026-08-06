#pragma once
#include "tensor.h"
#include <rfl.hpp>
#include <type_traits>
#include "primitives/parameter.h"

namespace Forge {
    template <typename T>
    concept HasParameters = requires(T x){
        {x.parameters()}->std::same_as<std::vector<Parameter>>;
    };
    template <typename T>
    std::vector<Parameter> extract_parameters(T& model);
}

template<typename T>
std::vector<Forge::Parameter> Forge::extract_parameters(T& model) {
    auto data_members {rfl::to_view(model)};
    std::vector<Parameter> parameters;

    rfl::apply(
        [&](auto*... members) {
            ([&]() {
                if constexpr (HasParameters<std::remove_pointer_t<decltype(members)>>) {
                    for (std::vector<Parameter> params {members->parameters()}; auto p : params) {parameters.push_back(p);}
                    for (auto p : extract_parameters(members)) parameters.push_back(p);
                }
            }(), ...);
        }, data_members.values());
    return parameters;
}
