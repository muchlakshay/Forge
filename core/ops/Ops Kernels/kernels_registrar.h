#pragma once
#include "../Dispatcher/dispatcher.h"
#include "CPU/UTILITY_OPS_CPU.h"

namespace Forge {
    template <ValidEnum T>
    void register_all_kernels(Dispatcher<T>& dispatcher);
}

template <ValidEnum T>
inline void Forge::register_all_kernels(Dispatcher<T> &dispatcher) {
    register_utility_kernels(dispatcher);
}


inline void Forge::register_utility_kernels(Dispatcher<UtilityOps>& dispatcher) {
    static StorageBackendCPU storageBackendCPU{};
    static ConstantCPU constantCPU{};
    static InitializeCPU initializeCPU{};
    static PrintCPU printCPU{};
    static StorageCopyCPU storageCopyCPU{};
    static RandomCPU randomCPU{};

    dispatcher.register_kernel<StorageBackend>(&storageBackendCPU, UtilityOps::storage_backend, DispatchKey::CPU);
    dispatcher.register_kernel<ConstantAbstract>(&constantCPU, UtilityOps::constant, DispatchKey::CPU);
    dispatcher.register_kernel<InitializeAbstract>(&initializeCPU, UtilityOps::initializers, DispatchKey::CPU);
    dispatcher.register_kernel<PrintAbstract>(&printCPU, UtilityOps::print, DispatchKey::CPU);
    dispatcher.register_kernel<StorageCopyAbstract>(&storageCopyCPU, UtilityOps::storage_copy, DispatchKey::CPU);
    dispatcher.register_kernel(&randomCPU, UtilityOps::random, DispatchKey::CPU);
}