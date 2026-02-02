#pragma once
#include <string_view>

#if defined(__GNUC__) || defined(__clang__)
#define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif  defined(__MSC_VER__)
#define CURRENT_FUNTION __FUNCSIG__
#else
#error "unsupported compiler"
#endif

class Kernel {
    constexpr std::string_view m_kernelName{};
    constexpr explicit Kernel(const std::string_view kernelName) : m_kernelName(kernelName) {}
    constexpr auto kernel_name_str() const {return m_kernelName;}
};