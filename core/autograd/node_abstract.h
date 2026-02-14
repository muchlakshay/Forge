#pragma once

class NodeAbstract {
    public:
    virtual void backward() const = 0;
    virtual ~NodeAbstract() = default;
};