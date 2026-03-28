#include "weights_init_cpu.h"

void Forge::WeightsInitCPU::initialize(Tensor& weights, std::size_t input_size, std::size_t output_size,
    Initializers initializer) const {
    std::random_device device;
    std::mt19937_64 mt{ device() };

    auto fill {
        [&](auto& storage, auto& dist) {
            DISPATCH_ALL_TYPES(weights.dtype(), Device::CPU,
                [&] {
                    auto* casted_ptr {static_cast<scalar_t*>(weights.data())};
                    for (std::size_t i {}; i<weights.size(); ++i)
                        casted_ptr[i] = static_cast<scalar_t>(dist(mt));
                });}
    };

    if (initializer == Initializers::xavier_normal) {
        std::normal_distribution<> distribution(0, std::sqrt(1.0 / input_size));
        fill(weights, distribution);
    } else if (initializer == Initializers::xavier_uniform) {
        const auto limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution distribution(-limit, limit);
        fill(weights, distribution);
    } else if (initializer == Initializers::he_normal) {
        std::normal_distribution<> distribution(0, std::sqrt(2.0 / input_size));
        fill(weights, distribution);
    } else if (initializer == Initializers::he_uniform) {
        auto limit = std::sqrt(6.0 / input_size);
        std::uniform_real_distribution distribution(-limit, limit);
        fill(weights, distribution);
    } else throw std::invalid_argument("Invalid initializer Type");
}


