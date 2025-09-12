#include "node.hpp"
#include "Tensor.hpp"
#include "ops_def.hpp"
#include <functional>
#include <vector>

#pragma once
template<> inline void Node<Tensor>::zero_grad(){
    this->grad = Tensor(0.0,this->value.shape());
}

template <> class Pow<Tensor> : public Node<Tensor> {
    double n;
public:
    Pow(Node<Tensor>* x, double n_) : n(n_)  { 
        this->input_size = 1;
        this->inputs = new Node<Tensor>*[1]{x};
        this->require_grad = this->inputs[0]->require_grad;
    }
    void forward() override {
        Tensor& x = this->inputs[0]->value;
        this->value = Tensor(x.shape());
        size_t sz = this->value.size();
        for(size_t i = 0;i < sz;i++){
            this->value[i] = std::pow(x[i], n);
        }
    }
    void backward() override {
        if(this->inputs[0]->require_grad){
            Tensor& x = this->inputs[0]->value;
            size_t sz = x.size();
            for(size_t i = 0;i < sz;i++){
                this->inputs[0]->grad[i] += this->grad[i] * n * std::pow(x[i], n - 1.0);
            }
        }
    }
};

template <class F> class UnaryFunc<Tensor, F> : public Node<Tensor> {
private:
    F elemwise_F;
public:
    UnaryFunc(Node<Tensor>* x) { 
        this->input_size = 1;
        this->inputs = new Node<Tensor>*[1]{x};
        this->require_grad = this->inputs[0]->require_grad;
    }
    void forward() override {
        this->value = Tensor(this->inputs[0]->value.shape()); 
        size_t sz = inputs[0]->value.size();
        for(size_t i = 0; i < sz;i++){
            this->value[i] = elemwise_F.forward(this->inputs[0]->value[i]);
        }
    }
    void backward() override {
        if(this->inputs[0]->require_grad){
            Tensor& x = this->inputs[0]->value;
            Tensor& fx = this->value;
            Tensor& grads = this->inputs[0]->grad;
            size_t sz = inputs[0]->value.size();
            for(size_t i = 0; i < sz;i++){
                grads[i] += this->grad[i] * elemwise_F.derivative(x[i],fx[i]);
            }
        }
    }
};

namespace ReduceType{
    struct Max{
        constexpr double forward(double& other, double& curr) const{
            return std::max(other, curr);
        }
        constexpr double backward(double& other, double& curr) const{
            return (other == curr) ? 1.0 : 0.0;
        }
    };
    struct Min{
        constexpr double forward(double& other, double& curr) const{
            return std::min(other, curr);
        }
        constexpr double backward(double& other, double& curr) const{
            return (other == curr) ? 1.0 : 0.0;
        }
    };
    struct Sum{
        constexpr double forward(double& other, double& curr) const{
            return other + curr;
        }
        constexpr double backward(double& other, double& curr) const{
            return 1.0;
        }
    };
}

namespace TensorOps{
    template <class F>  
    class Reduce : public Node<Tensor> {
    private:
        F elemwise_F;
    public:
        Reduce(Node<Tensor>* x) { 
            this->input_size = 1;
            this->inputs = new Node<Tensor>*[1]{x};
            this->require_grad = this->inputs[0]->require_grad;
        }
        void forward() override { 
            this->value = Tensor({1});
            size_t sz = inputs[0]->value.size();
            for(size_t i = 0; i < sz;i++){
                this->value[0] = elemwise_F.forward(inputs[0]->value[i],this->value[0]);
            }
        }
        void backward() override { 
            if(this->inputs[0]->require_grad){
                Tensor& x = this->inputs[0]->value;
                Tensor& grads = this->inputs[0]->grad;
                size_t sz = inputs[0]->value.size();
                for(size_t i = 0; i < sz;i++){
                    grads[i] += this->grad[i] * elemwise_F.derivative(x[i],this->value[0]); 
                }
            }
        }
    };

    class Conv2d : public Node<Tensor> {
    private:
        size_t _stride, _pad, _dilation;
    public:
        Conv2d(Node<Tensor>* x, Node<Tensor>* kernel, size_t stride, size_t pad, size_t dilation) : 
            _stride(stride), _pad(pad), _dilation(dilation)  { 
            this->input_size = 2;
            this->inputs = new Node<Tensor>*[2]{x, kernel};
            this->require_grad = this->inputs[0]->require_grad || this->inputs[1]->require_grad;
        }
        void forward() override {
            this->value = conv2d(this->inputs[0]->value,this->inputs[1]->value,
                this->_stride, this->_pad, this->_dilation);
        }
        void backward() override {
            if(this->inputs[0]->require_grad){
                Tensor dx = conv2d_dx(this->inputs[0]->value,this->inputs[1]->value, this->grad,
                    this->_stride, this->_pad, this->_dilation);
                this->inputs[0]->grad += dx;
            }
            if(this->inputs[1]->require_grad){
                Tensor dw = conv2d_dw(this->inputs[0]->value,this->inputs[1]->value, this->grad,
                    this->_stride, this->_pad, this->_dilation);
                this->inputs[1]->grad += dw;
            }
        }
    };

    class MatMul : public Node<Tensor> {
    public:
        MatMul(Node<Tensor>* a, Node<Tensor>* b)  { 
            this->input_size = 2;
            this->inputs = new Node<Tensor>*[2]{a, b};
            this->require_grad = this->inputs[0]->require_grad || this->inputs[1]->require_grad;
        }
        void forward() override {
            this->value = this->inputs[0]->value.matmul(this->inputs[1]->value);
        }
        void backward() override {
            if(this->inputs[0]->require_grad){
                Tensor t = this->inputs[1]->value.transpose();
                Tensor dA = this->grad.matmul(t);
                this->inputs[0]->grad += dA;
            }
            if(this->inputs[1]->require_grad){
                Tensor dB = this->inputs[0]->value.transpose().matmul(this->grad);
                this->inputs[1]->grad += dB;
            }
        }
    };

    class MaxPool2d : public Node<Tensor> {
    private:
        size_t _kH, _kW, _stride, _pad;
        std::vector<size_t> mask;
    public:
        MaxPool2d(Node<Tensor>* x,size_t kH, size_t kW,size_t stride, size_t pad) :
            _kH(kH), _kW(kW), _stride(stride), _pad(pad) { 
            this->input_size = 1;
            this->inputs = new Node<Tensor>*[2]{x};
            this->require_grad = this->inputs[0]->require_grad;
        }
        void forward() override {
            this->value = maxpool2d(this->inputs[0]->value,_kW,_kH,mask,_stride,_pad);
        }
        void backward() override {
            if(this->inputs[0]->require_grad){
                maxpool2d_backward(this->inputs[0]->value, this->grad, this->inputs[0]->grad,
                                    mask, _kW, _kH, _stride, _pad);
            }
        }
    };

    class Reshape : public Node<Tensor> {
    private:
        std::vector<size_t> output_shape;
    public:
        Reshape(Node<Tensor>* x, std::vector<size_t>&& shape)  { 
            this->input_size = 1;
            this->inputs = new Node<Tensor>*[1]{x};
            this->output_shape = shape;
            this->require_grad = this->inputs[0]->require_grad;
        }
        void forward() override {
            this->value = this->inputs[0]->value.reshape(std::move(this->output_shape));
        }
        void backward() override {
            if(this->inputs[0]->require_grad){
                Tensor t = this->grad.reshape(this->inputs[0]->value.shape());
                this->inputs[0]->grad += t;
            }
        }
    };
}
