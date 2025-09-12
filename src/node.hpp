#include <vector>
// ================== 基础节点定义 ==================
#pragma once
template<typename nT> class AutoGradGraph;

template<typename T>
class Node {
protected:
    Node<T>** inputs;
    size_t input_size = 0;
    size_t id;
public:
    friend class AutoGradGraph<T>;
    T value, grad;
    bool require_grad = true;
    void zero_grad();
    virtual ~Node() = default;
    virtual void forward() = 0;
    virtual void backward() = 0;

    Node() = default;
    Node(Node<T>&) noexcept = delete;
    Node(Node<T>&&) noexcept = delete;
    Node<T>& operator=(Node<T>&) noexcept = delete;
    Node<T>& operator=(Node<T>&&) noexcept = delete;
};

template<typename T>
class Variable :public Node<T> {
public:
    Variable(T v, bool require_grad = true) { 
        this->input_size = 0;
        this->value = v; 
        this->require_grad = require_grad;
    }
    Variable(T&& v, bool require_grad = true) { 
        this->input_size = 0;
        this->value = v; 
        this->require_grad = require_grad;
    }
    void forward() override {}
    void backward() override {}
};