#include "node.hpp"
#include "ops_def.hpp"
#pragma once
template<> inline void Node<double>::zero_grad(){
    this->grad = 0.0;
}

template <> class Pow<double> : public Node<double> {
    double n;
public:
    Pow(Node<double>* x, double n_) : n(n_)  { 
        this->input_size = 1;
        this->inputs = new Node<double>*[1]{x};
        this->require_grad = this->inputs[0]->require_grad;
    }
    void forward() override { this->value = std::pow(inputs[0]->value, n); }
    void backward() override {
        if(this->inputs[0]->require_grad){
            double x = this->inputs[0]->value;
            this->inputs[0]->grad += this->grad * n * std::pow(x, n - 1.0);
        }
    }
};

template <class F> class UnaryFunc<double, F> : public Node<double> {
private:
    F unaryF;
public:
    UnaryFunc(Node<double>* x) { 
        this->input_size = 1;
        this->inputs = new Node<double>*[1]{x};
        this->require_grad = this->inputs[0]->require_grad;
    }
    void forward() override { this->value = unaryF.forward(this->inputs[0]->value); }
    void backward() override { 
        if(this->inputs[0]->require_grad){
            this->inputs[0]->grad += this->grad * unaryF.derivative(this->inputs[0]->value,this->value); 
        }
    }
};