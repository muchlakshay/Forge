#include "weights_init_cpu.h"

void Forge::WeightsInitCPU::initialize(Tensor& weights, const std::vector<std::size_t>& dims,
                                       Initializers initializer) const {
    std::random_device device;
    std::mt19937_64 mt{ device() };
    auto cols {dims.back()};
    auto rows {dims[dims.size() - 2]};

    auto fill {
        [&](auto& W, auto& dist) {
            DISPATCH_ALL_TYPES(weights.dtype(), Device::CPU,
                [&] {
                    auto* casted_ptr {static_cast<scalar_t*>(W.data())};
                    for (std::size_t i {}; i<W.size(); ++i)
                        casted_ptr[i] = static_cast<scalar_t>(dist(mt));
                });}
    };

    if (initializer == Initializers::xavier_normal) {
        std::normal_distribution<> distribution(0, std::sqrt(1.0 / static_cast<double>(cols)));
        fill(weights, distribution);
    } else if (initializer == Initializers::xavier_uniform) {
        const auto limit = std::sqrt(6.0 / static_cast<double>(cols + rows));
        std::uniform_real_distribution distribution(-limit, limit);
        fill(weights, distribution);
    } else if (initializer == Initializers::he_normal) {
        std::normal_distribution<> distribution(0, std::sqrt(2.0 / static_cast<double>(cols)));
        fill(weights, distribution);
    } else if (initializer == Initializers::he_uniform) {
        auto limit = std::sqrt(6.0 / static_cast<double>(cols));
        std::uniform_real_distribution distribution(-limit, limit);
        fill(weights, distribution);
    } else throw std::invalid_argument("Invalid initializer Type");
}


