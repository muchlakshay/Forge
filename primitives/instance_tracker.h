#pragma once
#include <atomic>

namespace Forge {
    struct InstanceTracker {
        int m_count{0};

        InstanceTracker() = default;
        InstanceTracker(const InstanceTracker&) { ++m_count; }
        InstanceTracker(InstanceTracker&&) noexcept { ++m_count; }
        InstanceTracker& operator=(const InstanceTracker&) { ++m_count; return *this;}
        InstanceTracker& operator=(InstanceTracker&&) noexcept { ++m_count; return *this;}
    };
}