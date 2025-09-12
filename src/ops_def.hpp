#include <cmath>
#include "node.hpp"
#pragma once
// ================== 基础节点类型 ==================
template<typename T>
class Add : public Node<T> {
public:
    Add(Node<T>* a, Node<T>* b) { 
        this->input_size = 2;
        this->inputs = new Node<T>*[2]{a, b};
        this->require_grad = this->inputs[0]->require_grad || this->inputs[1]->require_grad;
    }
    void forward() override { this->value = this->inputs[0]->value + this->inputs[1]->value; }
    void backward() override {
        if(this->inputs[0]->require_grad){
            this->inputs[0]->grad += this->grad;
        }
        if(this->inputs[1]->require_grad){
            this->inputs[1]->grad += this->grad;
        }
    }
};
template<typename T>
class Sub : public Node<T> {
public:
    Sub(Node<T>* a, Node<T>* b) { 
        this->input_size = 2;
        this->inputs = new Node<T>*[2]{a, b};
        this->require_grad = this->inputs[0]->require_grad || this->inputs[1]->require_grad;
    }
    void forward() override { this->value = this->inputs[0]->value - this->inputs[1]->value; }
    void backward() override {
        if(this->inputs[0]->require_grad){
            this->inputs[0]->grad += this->grad;
        }
        if(this->inputs[1]->require_grad){
            this->inputs[1]->grad -= this->grad;
        }
    }
};

template<typename T>
class Mul : public Node<T> {
public:
    Mul(Node<T>* a, Node<T>* b) { 
        this->input_size = 2;
        this->inputs = new Node<T>*[2]{a, b};
        this->require_grad = this->inputs[0]->require_grad || this->inputs[1]->require_grad;
    }
    void forward() override { this->value = this->inputs[0]->value * this->inputs[1]->value; }
    void backward() override {
        if(this->inputs[0]->require_grad){
            this->inputs[0]->grad += this->grad * this->inputs[1]->value;
        }
        if(this->inputs[1]->require_grad){
            this->inputs[1]->grad += this->grad * this->inputs[0]->value;
        }
    }
};

template<typename T>
class Div : public Node<T> {
public:
    Div(Node<T>* a, Node<T>* b) { 
        this->input_size = 2;
        this->inputs = new Node<T>*[2]{a, b};
        this->require_grad = this->inputs[0]->require_grad || this->inputs[1]->require_grad;
    }
    void forward() override { this->value = this->inputs[0]->value / this->inputs[1]->value; }
    void backward() override {
        if(this->inputs[0]->require_grad){
            this->inputs[0]->grad += this->grad / this->inputs[1]->value;
        }
        if(this->inputs[1]->require_grad){
            T delta = this->inputs[0]->value;
            delta /=  this->inputs[1]->value;
            delta /=  this->inputs[1]->value;
            delta *= this->grad;
            this->inputs[1]->grad -= delta;
            //this->inputs[1]->grad += -this->grad * this->inputs[0]->value / this->inputs[1]->value / this->inputs[1]->value;
        }
    }
};
// ================== 特殊函数节点类型 ==================
template <typename T> class Pow : public Node<T>{};
template <typename T, class F> class UnaryFunc : public Node<T>{};
struct SinOp{
    constexpr double forward(double& x) const{ return std::sin(x); }
    constexpr double derivative(double& x, double& fx) const{ return std::cos(x); }
};
struct CosOp{
    constexpr double forward(double& x) const{ return std::cos(x); }
    constexpr double derivative(double& x, double& fx) const{ return -std::sin(x); }
};

struct ExpOp{
    constexpr double forward(double& x) const{ return std::exp(x); }
    constexpr double derivative(double& x, double& fx) const{ return fx; }
};

struct LogOp{
    constexpr double forward(double& x) const{ return std::log(x); }
    constexpr double derivative(double& x, double& fx) const{ return 1.0 / x; }
};

struct ReLUOp{
    constexpr double forward(double& x) const{ return std::max(x ,0.0); }
    constexpr double derivative(double& x, double& fx) const{ return x > 0.0 ? 1.0 : 0.0; }
};

struct SigmoidOp{
    constexpr double forward(double& x) const{ 
        if(x >= 0.0){
            double e = std::exp(-x);
            return 1.0 / (1.0 + e);
        }else{
            double e = std::exp(x);
            return e / (1.0 + e);
        }
    }
    constexpr double derivative(double& x, double& fx) const{ return fx*(1.0-fx); }
};

struct TanhOp{
    constexpr double forward(double& x) const{ 
        if(x >= 0.0){
            double e = std::exp(-2*x);
            return (1.0 - e) / (e + 1.0);
        }else{
            double e = std::exp(2*x);
            return (e - 1.0) / (e + 1.0);
        }
    }
    constexpr double derivative(double& x, double& fx) const{ return 1.0 - fx*fx; }
};