#pragma once
#include <cstddef>

namespace Forge {
    template <typename DTYPE>
   void broadcastADD_AVX2(DTYPE* A, DTYPE* x, DTYPE* B, std::size_t rows, std::size_t cols);
}

template<typename DTYPE>
void Forge::broadcastADD_AVX2(DTYPE *A, DTYPE *x, DTYPE* B, std::size_t rows, std::size_t cols) {
    constexpr bool isFloat {std::is_same_v<std::decay_t<DTYPE>, float>};
    constexpr bool isDouble {std::is_same_v<std::decay_t<DTYPE>, double>};
    constexpr std::size_t step {isFloat ? 8 : (isDouble ? 4 : 1)};
    std::size_t process_cols {4};
    std::size_t SIMD_steps {(cols / (step * process_cols))};

    if (SIMD_steps!=0) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < rows; i++) {
            auto r_A {A+i*cols}, r_B{B+i*cols};
            for (std::size_t j = 0; j<SIMD_steps; j++) {
                auto col_start {j*step*process_cols};
                if constexpr (isFloat) {
                    _mm256_storeu_ps(r_B+col_start,        _mm256_add_ps(   _mm256_load_ps( r_A+col_start   ), _mm256_loadu_ps( x + col_start ) )      );
                    _mm256_storeu_ps(r_B+col_start  +step, _mm256_add_ps(_mm256_load_ps(r_A+col_start+step  ), _mm256_loadu_ps( x + col_start + step  )));
                    _mm256_storeu_ps(r_B+col_start+2*step, _mm256_add_ps(_mm256_load_ps(r_A+col_start+2*step), _mm256_loadu_ps( x + col_start + 2*step)));
                    _mm256_storeu_ps(r_B+col_start+3*step, _mm256_add_ps(_mm256_load_ps(r_A+col_start+3*step), _mm256_loadu_ps( x + col_start + 3*step)));

                } else if constexpr (isDouble) {
                    _mm256_storeu_pd(r_B+col_start,        _mm256_add_pd(   _mm256_load_pd( r_A+col_start   ), _mm256_loadu_pd( x + col_start ) )      );
                    _mm256_storeu_pd(r_B+col_start  +step, _mm256_add_pd(_mm256_load_pd(r_A+col_start+step  ), _mm256_loadu_pd( x + col_start + step  )));
                    _mm256_storeu_pd(r_B+col_start+2*step, _mm256_add_pd(_mm256_load_pd(r_A+col_start+2*step), _mm256_loadu_pd( x + col_start + 2*step)));
                    _mm256_storeu_pd(r_B+col_start+3*step, _mm256_add_pd(_mm256_load_pd(r_A+col_start+3*step), _mm256_loadu_pd( x + col_start + 3*step)));
                }
            }
        }

        if (std::size_t SIMD_cols{SIMD_steps*step*process_cols}; SIMD_cols < cols) {
            #pragma omp parallel for schedule(static)
            for (std::size_t i=0; i<rows; ++i) {
                auto r_A {A+i*cols}, r_B{B+i*cols};
                for (std::size_t j {SIMD_cols}; j<cols; ++j) {
                    r_B[j] = r_A[j]+x[j];
                }
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t i=0; i<rows; ++i) {
        auto r_A {A+i*cols}, r_B{B+i*cols};
        for (std::size_t j =0; j<cols; ++j) {
            r_B[j] = r_A[j]+x[j];
        }
    }
}