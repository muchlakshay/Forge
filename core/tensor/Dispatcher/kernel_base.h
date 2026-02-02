#pragma once
#include <source_location>
#include <string_view>
#include <ctti/type_id.hpp>

namespace Forge {
    class Kernel;
}

class Forge::Kernel {
    ctti::type_id_t m_derived_id{};
public:
    constexpr explicit Kernel(const ctti::type_id_t& derived_id) : m_derived_id{derived_id} {}
    [[nodiscard]] constexpr ctti::type_id_t derived_id() const {return m_derived_id;}
};