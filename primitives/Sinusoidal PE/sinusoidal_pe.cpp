#include "sinusoidal_pe.h"
#include <cmath>

Forge::Tensor Forge::sinusoidal_pe(std::size_t seq_len, std::size_t d_model,
    Dtype dtype, Device device) {
    Tensor encoding {{seq_len, d_model}, dtype, false, device};
    double log10k {std::log(10000.0)};
    double one_by_dmodel {1.0/static_cast<double>(d_model)};

    std::vector<double> divisors (d_model/2);
    for (std::size_t i {}; i<d_model; i+=2) divisors[i/2] = std::exp(-2.0 * i * one_by_dmodel * log10k);

    DISPATCH_ALL_TYPES(dtype, device, [&] {
        auto encoding_map {encoding.as_eigen<scalar_t, 2>()};

        for (std::size_t pos {}; pos<seq_len; ++pos) {
            auto pos_casted {static_cast<scalar_t>(pos)};
            for (std::size_t i {}; i<d_model; i+=2) {
                 auto scaled_pos {pos_casted * static_cast<scalar_t>(divisors[i/2])};
                 encoding_map(pos, i) = Eigen::numext::sin(scaled_pos);
                 if (i+1 < d_model ) encoding_map(pos, i+1) = Eigen::numext::cos(scaled_pos);
            }
        }
    });

    return encoding;
}
