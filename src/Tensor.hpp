#include <vector>
#include <stdlib.h>
#include <stdexcept>
#include "node.hpp"

#pragma once
#define CHECK_TENSOR_SHAPE(other)\
    do { \
        if(this->numel != other.numel || this->_dims != other._dims) \
            throw std::length_error("Tensor shape not match"); \
    } while (0)

class Tensor{
private:
    double* _data = nullptr;
    size_t numel = 0;
    std::vector<size_t> _dims;
    std::vector<size_t> strides;
    
    void _init(const std::vector<size_t>& dims);
    void _swap(Tensor& _other);
    void _update_stride();
    size_t ravel_idx(const std::vector<size_t>& indices) const;
public:
    Tensor(){this->numel = 0;}
    Tensor(const std::vector<size_t>& dims);
    Tensor(double val, const std::vector<size_t>& dims);
    Tensor(std::vector<size_t>&& _dims);
    Tensor(double val, std::vector<size_t>&& _dims);

    Tensor(std::vector<std::vector<double>> mat);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);
    ~Tensor();

    double operator[](size_t i) const { 
        // if(i >= this->numel)
        //     throw std::out_of_range("operator[] out of range for tensor len=" 
        //         + std::to_string(this->numel));
        return _data[i];
    }
    double& operator[](size_t i) { 
        // if(i >= this->numel) 
        //     throw std::out_of_range("operator[] out of range for tensor len=" 
        //         + std::to_string(this->numel));
        return _data[i];
    }
    
    template<typename... Indices>
    double& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) > 0, "At least one index required");
        return this->_data[ravel_idx({static_cast<size_t>(indices)...})];
    }
    
    template<typename... Indices>
    double operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) > 0, "At least one index required");
        return this->_data[ravel_idx({static_cast<size_t>(indices)...})];
    }

    std::vector<size_t> shape() const { return this->_dims; };
    void fill(double val);
    size_t size() const{ return this->numel; }
    void unsqueeze(int i){ 
        if(i < 0){
            this->_dims.insert(this->_dims.end()+i,1LL);
        }else{
            this->_dims.insert(this->_dims.begin()+i,1LL);
        }
        this->_update_stride();
    }

    Tensor matmul(Tensor& other);
    Tensor transpose();
    Tensor reshape(std::vector<size_t>&& sz){
        Tensor output = Tensor(sz);
        if(output.numel != this->numel){
            throw std::runtime_error("Reshape size not match");
        }
        memcpy(output._data, this->_data, this->numel*sizeof(double));
    }
    
    bool operator==(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        Tensor res = Tensor(*this);
        for(int i = 0;i < this->numel;i++){
            if(res._data[i] != other._data[i]) return false;
        }
        return true;
    }

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);

    bool operator!=(Tensor& other){
        return !(*this == other);
    }

    Tensor operator+(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        Tensor res = Tensor(*this);
        for(int i = 0;i < this->numel;i++){
            res._data[i] += other._data[i];
        }
        return res;
    }
    void operator-(){
        for(int i = 0;i < this->numel;i++){
            this->_data[i] = -this->_data[i];
        }
    }
    Tensor operator-(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        Tensor res = Tensor(*this);
        for(int i = 0;i < this->numel;i++){
            res._data[i] -= other._data[i];
        }
        return res;
    }
    Tensor operator*(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        Tensor res = Tensor(*this);
        for(int i = 0;i < this->numel;i++){
            res._data[i] *= other._data[i];
        }
        return res;
    }
    Tensor operator/(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        Tensor res = Tensor(*this);
        for(int i = 0;i < this->numel;i++){
            res._data[i] /= other._data[i];
        }
        return res;
    }

    void operator+=(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        for(int i = 0;i < this->numel;i++){
            this->_data[i] += other._data[i];
        }
    }
    void operator-=(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        for(int i = 0;i < this->numel;i++){
            this->_data[i] -= other._data[i];
        }
    }
    void operator*=(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        for(int i = 0;i < this->numel;i++){
            this->_data[i] *= other._data[i];
        }
    }
    void operator/=(Tensor& other){
        CHECK_TENSOR_SHAPE(other);
        for(int i = 0;i < this->numel;i++){
            this->_data[i] /= other._data[i];
        }
    }
};

Tensor conv2d(const Tensor& x, const Tensor& w,
                  size_t stride  = 1,
                  size_t pad     = 0,
                  size_t dilation = 1);
Tensor conv2d_dw(const Tensor& x, const Tensor& w, const Tensor& dy,
                                     size_t stride   = 1,
                                     size_t pad      = 0,
                                     size_t dilation = 1);

Tensor conv2d_dx(const Tensor& x, const Tensor& w, const Tensor& dy,
                                     size_t stride   = 1,
                                     size_t pad      = 0,
                                     size_t dilation = 1);

Tensor maxpool2d(const Tensor& x, 
    size_t kH, size_t kW,
    std::vector<size_t>& mask,
    size_t stride = 1, size_t pad = 0);

void maxpool2d_backward(const Tensor& x,
                          const Tensor& dy,
                          Tensor& dx,
                          const std::vector<size_t>& mask,
                          size_t kH, size_t kW,size_t stride = 1, size_t pad = 0);