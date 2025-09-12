#include <vector>
#include <stdlib.h>
#include <cstring>
#include <stdexcept>
#include "Tensor.hpp"
#include <limits>


void Tensor::_swap(Tensor& _other){
    std::swap(this->_data, _other._data);
    std::swap(this->numel, _other.numel);
    std::swap(this->_dims, _other._dims);
    std::swap(this->strides, _other.strides);
}

void Tensor::_update_stride(){
    strides.resize(this->_dims.size());
    strides.back() = 1;
    size_t sz = this->_dims.size() - 1;
    for (size_t i = 0; i < sz; i++) {
        strides[i] = strides[i + 1] * this->_dims[i + 1];
    }
}

void Tensor::_init(const std::vector<size_t>& dims){
    if(dims.size() == 0){
        this->_dims = {1};
        this->numel = 1;
    }else{
        this->_dims = dims;
        this->numel = dims[0];
        for(size_t i = 1;i < dims.size();i++){
            this->numel *= dims[i];
        }
        if(this->numel == 0) return;
    }
    this->_data =  (double*)malloc(this->numel*sizeof(double));
    if(this->_data == nullptr){
        this->numel = 0;
        this->_dims.clear();
        throw std::bad_alloc();
    }
    this->_update_stride();
}

size_t Tensor::ravel_idx(const std::vector<size_t>& indices) const{
    if(indices.size() != this->_dims.size()) {
        throw std::out_of_range("Number of indices doesn't match tensor dimension");
    }
    if(indices.size() != this->strides.size()){
        throw std::out_of_range("stride not match " + std::to_string(indices.size()) + "!=" + std::to_string(this->strides.size()));
    }
    
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= this->_dims[i]) {
            throw std::out_of_range("Index " + std::to_string(indices[i]) + " out of bounds");
        }
        index += indices[i] * strides[i];
    }
    if(index >= this->numel){
        throw std::out_of_range("Index " + std::to_string(index) + " out of bounds for _data");
    }
    return index;
}

void Tensor::fill(double val){
    if(this->_data == nullptr){
        return;
    }
    for(size_t n = 0; n < this->numel;n++){
        this->_data[n] = val;
    }
}

Tensor::Tensor(double val, const std::vector<size_t>& _dims){ _init(_dims); fill(val); }

Tensor::Tensor(const std::vector<size_t>& dims){ _init(dims); }

Tensor::Tensor(double val, std::vector<size_t>&& _dims){
    std::vector<size_t> dim = _dims;
    _init(dim);
    fill(val);
}

Tensor::Tensor(std::vector<size_t>&& _dims){
    std::vector<size_t> dim = _dims;
    _init(dim);
}

Tensor::Tensor(std::vector<std::vector<double>> mat){
    if(mat.size() == 0) return;
    size_t r = mat.size(), c = mat[0].size();
    std::vector<size_t> dims = {r, c};
    _init(dims);
    for(size_t i = 0;i < r;i++){
        for(size_t j = 0;j < c;j++){
            this->_data[i*r+j] = mat[i][j];
        }
    }
}


Tensor::Tensor(const Tensor& other){
    this->numel = other.numel;
    this->_dims = other._dims;
    if(this->_data != nullptr){
        double* new_data = (double*)realloc(this->_data,this->numel*sizeof(double));
        if(new_data == nullptr){
            this->numel = 0;
            free(this->_data);
            this->_dims.clear();
            throw std::bad_alloc();
            return;
        }
        this->_data = new_data;
    }else{
        this->_data = (double*)malloc(this->numel*sizeof(double));
        if(this->_data == nullptr){
            this->numel = 0;
            this->_dims.clear();
            throw std::bad_alloc();
            return;
        }
    }
    memcpy(this->_data,other._data,this->numel*sizeof(double));
    this->_update_stride();
}

Tensor::Tensor(Tensor&& other){
    this->numel = other.numel;
    this->_dims = other._dims;
    this->_data =  other._data;
    this->strides = other.strides;
    other.numel = 0;
    other._dims.clear();
    other.strides.clear();
    other._data = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other){
    if(this != &other){
        Tensor(std::move(other))._swap(*this);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other){
    if(this != &other){
        Tensor(other)._swap(*this);
    }
    return *this;
}

Tensor::~Tensor(){
    if(this->_data != nullptr)
        free(this->_data);
};

void _matrix_mul(double* A, size_t rA, size_t cA,
                double* B, size_t rB, size_t cB, double* out){
    if(cA != rB) 
        throw std::range_error("Matrix size not match");
    memset(out,0,sizeof(double)*rA*cB);
    for (size_t i = 0; i < rA; i++){
        for (size_t j = 0; j < cB; j++){
            for (size_t k = 0; k < cA; k++){
                out[i*cB+j] += A[i*cA+k]*B[k*cB+j];
            }
            
        }
    }
}

Tensor Tensor::matmul(Tensor& other){
    size_t dimA = this->_dims.size();
    size_t dimB = other._dims.size();
    if(dimA < 2 || dimA != dimB){
        throw std::range_error("Tensor cannot view as matrix");
    }else{
        for (size_t i = 0; i < dimA - 2; ++i)
            if (this->_dims[i] != other._dims[i])
                throw std::invalid_argument("batch shape mismatch");
        size_t rA = this->_dims[dimA-2], cA = this->_dims[dimA - 1];
        size_t rB = other._dims[dimB-2], cB = other._dims[dimB -1];
        std::vector<size_t> out_shape(this->_dims.begin(), this->_dims.end() - 2);
        out_shape.push_back(rA); out_shape.push_back(cB);
        Tensor output(out_shape);

        size_t batch = 1;
        for (size_t i = 0; i < dimA - 2; ++i) 
            batch *= this->_dims[i];
        const size_t Amn = rA * cA, Bkn = rB * cB,  Cmn = rA * cB;
        for (size_t i = 0; i < batch; ++i) {
            _matrix_mul(this->_data + i* Amn, rA, cA,
                    other._data + i * Bkn, rB, cB,
                    output._data + i * Cmn);
        }
        return output;
    }
}

Tensor Tensor::transpose(){
    size_t dim_len = this->_dims.size();
    if(dim_len < 2){
        throw std::range_error("Tensor cannot view as matrix");
    }else{
        size_t rA = this->_dims[dim_len-2], cA = this->_dims[dim_len - 1];
        size_t batch = 1;
        for (size_t i = 0; i < dim_len - 2; ++i) 
            batch *= this->_dims[i];
        std::vector<size_t> output_shape = this->_dims;
        std::swap(output_shape[dim_len-1],output_shape[dim_len-2]);
        Tensor output(output_shape);
        const size_t mat_size = rA * cA;
        for (size_t b = 0; b < batch; ++b) {
            double* mat_batch_i = this->_data + b* mat_size;
            double* output_batch_i = output._data  + b* mat_size;
            for (size_t i = 0; i < rA; i++){
                for (size_t j = 0; j < cA; j++){
                    output_batch_i[j*rA+i] = mat_batch_i[i*cA+j];
                }
            }
        }
        return output;
    }
}


Tensor conv2d(const Tensor& x, const Tensor& w, size_t stride, size_t pad, size_t dilation){
    const auto& xsh = x.shape();   // N C H W
    const auto& wsh = w.shape();   // K C KH KW
    if (xsh.size() != 4 || wsh.size() != 4)
        throw std::invalid_argument("Conv2d: shape must be 4-D");
    if (xsh[1] != wsh[1])
        throw std::invalid_argument("Conv2d: input channel not match");

    const size_t N = xsh[0]; const size_t C = xsh[1];
    const size_t H = xsh[2]; const size_t W = xsh[3];

    const size_t K = wsh[0]; const size_t KH = wsh[2]; const size_t KW = wsh[3];

    const size_t OH = (H + 2 * pad - dilation * (KH - 1) - 1) / stride + 1;
    const size_t OW = (W + 2 * pad - dilation * (KW - 1) - 1) / stride + 1;

    Tensor out({N, K, OH, OW});

    for (size_t n = 0; n < N; ++n)
    for (size_t k = 0; k < K; ++k)
    for (size_t oh = 0; oh < OH; ++oh)
    for (size_t ow = 0; ow < OW; ++ow)
    {
        double sum = 0.0;
        for (size_t c = 0; c < C; ++c)
        for (size_t r = 0; r < KH; ++r)
        for (size_t s = 0; s < KW; ++s)
        {
            const size_t ih = oh * stride + r * dilation;
            const size_t iw = ow * stride + s * dilation;
            if (ih < pad || iw < pad) continue;
            const size_t ih0 = ih - pad;
            const size_t iw0 = iw - pad;
            if (ih0 >= H || iw0 >= W) continue;
            sum += x(n, c, ih0, iw0) * w(k, c, r, s);
        }
        out(n, k, oh, ow) = sum;
    }
    return out;
}


Tensor conv2d_dw(const Tensor& x, const Tensor& w, const Tensor& dy,
                                     size_t stride,
                                     size_t pad,
                                     size_t dilation){
    const auto& xsh = x.shape();
    const auto& wsh = w.shape();
    const auto& dysh = dy.shape();
    size_t N = xsh[0], C = xsh[1], H = xsh[2], W = xsh[3];
    size_t K = wsh[1], KH = wsh[2], KW = wsh[3];
    size_t OH = dysh[2], OW = dysh[3];

    Tensor dw({C, K, KH, KW});

    for (size_t c = 0; c < C; ++c)
      for (size_t k = 0; k < K; ++k)
        for (size_t kh = 0; kh < KH; ++kh)
          for (size_t kw = 0; kw < KW; ++kw) {
              double acc = 0.0;
              for (size_t n = 0; n < N; ++n)
                for (size_t oh = 0; oh < OH; ++oh)
                  for (size_t ow = 0; ow < OW; ++ow) {
                      size_t ih = oh * stride + kh * dilation;
                      size_t iw = ow * stride + kw * dilation;
                      if (ih < pad || iw < pad) continue;
                      ih -= pad;  iw -= pad;
                      if (ih >= H || iw >= W) continue;
                      acc += dy(n, k, oh, ow) * x(n, c, ih, iw);
                  }
              dw(c, k, kh, kw) = acc;
          }
    return dw;
}


Tensor conv2d_dx(const Tensor& x, const Tensor& w, const Tensor& dy,
                                     size_t stride,
                                     size_t pad,
                                     size_t dilation){
    const auto& xsh = x.shape();
    const auto& wsh = w.shape();
    const auto& dysh = dy.shape();
    size_t N = xsh[0], C = xsh[1], H = xsh[2], W = xsh[3];
    size_t K = wsh[1], KH = wsh[2], KW = wsh[3];
    size_t OH = dysh[2], OW = dysh[3];

    Tensor dx({N, C, H, W});

    for (size_t n = 0; n < N; ++n)
      for (size_t c = 0; c < C; ++c)
        for (size_t ih = 0; ih < H; ++ih)
          for (size_t iw = 0; iw < W; ++iw) {
              double acc = 0.0;
              for (size_t k = 0; k < K; ++k)
                for (size_t kh = 0; kh < KH; ++kh)
                  for (size_t kw = 0; kw < KW; ++kw) {
                      size_t delta_h = ih + pad;
                      size_t delta_w = iw + pad;
                      size_t off_h   = kh * dilation;
                      size_t off_w   = kw * dilation;
                      if (delta_h < off_h || delta_w < off_w) continue;
                      size_t rem_h = delta_h - off_h;
                      size_t rem_w = delta_w - off_w;
                      if (rem_h % stride != 0 || rem_w % stride != 0) continue;
                      size_t oh = rem_h / stride;
                      size_t ow = rem_w / stride;
                      if (oh >= OH || ow >= OW) continue;
                      acc += dy(n, k, oh, ow) * w(c, k, kh, kw);
                  }
              dx(n, c, ih, iw) = acc;
          }

    return dx;
}


Tensor maxpool2d(const Tensor& x, 
    size_t kH, size_t kW,
    std::vector<size_t>& mask,
    size_t stride, size_t pad){
    const auto& xsh = x.shape();   // N C H W
    size_t N = xsh[0], C = xsh[1], H = xsh[2], W = xsh[3];

    size_t OH = (H + 2*pad - kH) / stride + 1;;
    size_t OW = (W + 2*pad - kH) / stride + 1;
    if (OH == 0 || OW == 0)
        throw std::invalid_argument("output size == 0");

    Tensor y({N, C, OH, OW});
    mask.resize(N * C * OH * OW);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t max_idx = 0;

                    for (size_t kh = 0; kh < kH; ++kh) {
                        for (size_t kw = 0; kw < kW; ++kw) {
                            size_t ih = oh * stride + kh;
                            size_t iw = ow * stride + kw;
                            if (ih < pad || iw < pad) continue;
                            ih -= pad;  iw -= pad;
                            if (ih >= H || iw >= W) continue;

                            size_t flat = (n * C + c) * H * W + ih * W + iw;
                            double v = x(n, c, ih, iw);
                            if (v > max_val) {
                                max_val = v;
                                max_idx = flat;
                            }
                        }
                    }
                    y(n, c, oh, ow) = max_val;
                    mask[((n * C + c) * OH + oh) * OW + ow] = max_idx;
                }
            }
        }
    }
    return y;
}

void maxpool2d_backward(const Tensor& x,
                          const Tensor& dy,
                          Tensor& dx,
                          const std::vector<size_t>& mask,
                          size_t kH, size_t kW,size_t stride, size_t pad){
    const auto& dysh = dy.shape();
    size_t N = dysh[0], C = dysh[1];
    size_t H = dysh[2], W = dysh[3]; 

    size_t OH = (H + 2*pad - kH) / stride + 1;;
    size_t OW = (W + 2*pad - kH) / stride + 1;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    size_t dst_idx = ((n * C + c) * OH + oh) * OW + ow;
                    double dx_elem = dy(n, c, oh, ow);
                    size_t rem = mask[dst_idx] % (H * W);
                    size_t ih = rem / W, iw = rem % W;
                    dx(n, c, ih, iw) += dx_elem;
                }
            }
        }
    }
}