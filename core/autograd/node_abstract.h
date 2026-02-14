#pragma once

namespace Forge {
    class NodeAbstract;
}

class Forge::NodeAbstract {
    public:
    virtual void backward() const = 0;
    virtual ~NodeAbstract() = default;
};