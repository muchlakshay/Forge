#include "optimizers.h"
#include "../Dispatcher/primitive_dispatcher.h"
#include "optimizersUpdateAbstract.h"
#include <algorithm>

namespace Forge {
    void invariants_checks_rmv_duplicates(std::vector<Forge::Tensor *>& params);
}
void Forge::invariants_checks_rmv_duplicates(std::vector<Forge::Tensor *>& params) {
    if (params.empty()) throw std::invalid_argument("No parameters provided");
    const auto device {params[0]->device()};
    const auto dtype {params[0]->dtype()};
    for (const auto p : params) {
        if (p->device() != device) throw std::invalid_argument("parameters have different device");
        if (p->dtype() != dtype) {throw std::invalid_argument("parameters have different dtype");}
    };
    std::ranges::sort(params);
    const auto it {std::ranges::unique(params).begin()};
    params.erase(it, params.end());
}


