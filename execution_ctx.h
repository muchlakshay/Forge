#pragma once
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

}
