#pragma once
#define EIGEN_USE_THREADS
#define EIGEN_USE_BLAS
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Forge {
    struct ExecutionCtx {
        bool use_MultiThreading{};
        int threads {};
        Eigen::DefaultDevice default_device;
        std::unique_ptr<Eigen::ThreadPool> thread_pool {};
        std::unique_ptr<Eigen::ThreadPoolDevice> thread_pool_device {};

        ExecutionCtx() = default;
        ExecutionCtx(ExecutionCtx&) = delete;
        ExecutionCtx(ExecutionCtx&&) = delete;
        ExecutionCtx& operator=(ExecutionCtx&) = delete;
        ExecutionCtx& operator=(ExecutionCtx&&) = delete;

        void set_threads(int threads) {
            if (threads<=1) {use_MultiThreading=false; thread_pool.reset(); thread_pool_device.reset();}
            else{
                use_MultiThreading = true;
                thread_pool = std::make_unique<Eigen::ThreadPool>(threads);
                thread_pool_device = std::make_unique<Eigen::ThreadPoolDevice>(thread_pool.get(), threads);
            }
        }
    };

    inline ExecutionCtx global_exeCtx;

    inline float linear_time {};
    inline float attention_time{};
    inline float layernorm_time{};
    inline float embedding_time{};
    inline float broadcast_time{};
    inline float gelu_time {};

}

#define forge_eval(dest_map, eigen_expr) \
    do { \
        if (Forge::global_exeCtx.use_MultiThreading) { \
            (dest_map).device(*Forge::global_exeCtx.thread_pool_device) = (eigen_expr); \
        } else { \
            dest_map= eigen_expr; \
        } \
    } while (0)
