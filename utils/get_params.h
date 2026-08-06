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

