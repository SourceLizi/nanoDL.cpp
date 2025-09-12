#include "../src/Tensor.hpp"
#include <iostream>

void print_shape(const std::vector<size_t>& shape) {
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]";
}

bool test2Tensor(const std::string& name,
              const Tensor& out,
              const Tensor& expected)
{
    const auto& out_shape = out.shape();
    const auto& expected_shape = expected.shape();
    if (out_shape != expected_shape) {
        std::cout << name << " FAIL: shape mismatch";
        std::cout << ", expected "; print_shape(expected_shape);
        std::cout << ", but got "; print_shape(out_shape);
        return false;
    }
    bool ret = true;
    for (size_t n = 0; n < out_shape[0]; ++n)
    for (size_t k = 0; k < out_shape[1]; ++k)
    for (size_t h = 0; h < out_shape[2]; ++h)
    for (size_t w = 0; w < out_shape[3]; ++w)
        if (std::abs(out(n, k, h, w)-expected(n, k, h, w)) > 1e-8)  {
            std::cout << name << " FAIL: value mismatch at ("
                      << n << "," << k << "," << h << "," << w << "), expect " << 
                      expected(n, k, h, w)  <<" ,but got "<< out(n, k, h, w) <<"\n";
            ret = false;
        }
    if(ret){
        std::cout << name << " PASS\n";
    }
    return ret;
}

int base_test(){
        //张量生成
    Tensor tensor(0.0,{2,2});
    //大小确认
    if(tensor.shape()[0] != 2 || tensor.shape()[1] != 2){
        std::cout << "Check shape error" << std::endl;
        return -1;
    }
    //张量值确认
    for(int i = 0;i < 4; i++){
        if(tensor[0] != 0.0){
            std::cout << "Check Value error" << std::endl;
            return -1;
        }
    }
    //填充值确认
    tensor.fill(255.9);
    for(int i = 0;i < 4; i++){
        if(tensor[0] != 255.9){
            std::cout << "Check fill error" << std::endl;
            return -1;
        }
    }

    Tensor tensor1({{1,2},{3,4}});
    Tensor tensor2({{5,6},{7,8}});

    if(tensor1(0,0) != 1.0){
        std::cout << "Check index error" << std::endl;
        return -1;
    }
    return 0;
}
int ops_test(){
    
    Tensor tensor1({{1,2},{3,4}});
    Tensor tensor2({{5,6},{7,8}});
    
    Tensor tensor_add({{6,8},{10,12}});
    Tensor tensor_sub({{-4,-4},{-4,-4}});
    Tensor tensor_mul({{5,12},{21,32}});
    Tensor tensor_div({{1.0/5.0,2.0/6.0},{3.0/7.0,4.0/8.0}});
    
    //重载运算符测试
    if(tensor1 == tensor2){
        std::cout << "Check == error" << std::endl;
        return -1;
    }

    if((tensor1 + tensor2) != tensor_add){
        std::cout << "Check add error" << std::endl;
        return -1;
    }
    if((tensor1 - tensor2) != tensor_sub){
        std::cout << "Check sub error" << std::endl;
        return -1;
    }
    if((tensor1 * tensor2) != tensor_mul){
        std::cout << "Check mul error" << std::endl;
        return -1;
    }
    if((tensor1 / tensor2) != tensor_div){
        std::cout << "Check div error" << std::endl;
        return -1;
    }
    return 0;
}

int conv2d_test(){
    bool ok = true;

    //-------- 1. 1×1 卷积，stride=1, pad=0, dilate=1 --------
    {
        Tensor in({1, 1, 2, 2});
        Tensor ker({1, 1, 1, 1});
        in(0, 0, 0, 0) = 1; in(0, 0, 0, 1) = 2;
        in(0, 0, 1, 0) = 3; in(0, 0, 1, 1) = 4;
        ker(0, 0, 0, 0) = 5;

        Tensor expect({1, 1, 2, 2});
        for (int i = 0; i < 4; ++i) expect[i] = (i + 1) * 5;
        Tensor out = conv2d(in, ker, 1, 0, 1);
        ok &= test2Tensor("1x1 kernel", out, expect);
    }

    //-------- 2. 3×3 核，pad=1，stride=1，dilate=1 --------
    {
        Tensor in(1.0,{1, 1, 3, 3});
        Tensor ker(1.0,{1, 1, 3, 3});

        Tensor expect({1, 1, 3, 3});
        expect(0,0,0,0)=4; expect(0,0,0,1)=6; expect(0,0,0,2)=4;
        expect(0,0,1,0)=6; expect(0,0,1,1)=9; expect(0,0,1,2)=6;
        expect(0,0,2,0)=4; expect(0,0,2,1)=6; expect(0,0,2,2)=4;
        Tensor out = conv2d(in, ker,  1, 1, 1);
        ok &= test2Tensor("3x3 kernel pad1", out, expect);
    }

    //-------- 3. stride=2，pad=0，dilate=1 --------
    {
        Tensor in({1, 1, 4, 4});
        Tensor ker({1, 1, 2, 2});
        for (size_t i = 0; i < 16; ++i) in[i] = i + 1;
        for (size_t i = 0; i < 4; ++i) ker[i] = 1;

        Tensor expect({1, 1, 2, 2});
        expect(0, 0, 0, 0) = 1 + 2 + 5 + 6;
        expect(0, 0, 0, 1) = 3 + 4 + 7 + 8;
        expect(0, 0, 1, 0) = 9 + 10 + 13 + 14;
        expect(0, 0, 1, 1) = 11 + 12 + 15 + 16;
        Tensor out = conv2d(in, ker, 2, 0, 1);
        ok &= test2Tensor("stride=2", out, expect);
    }

    //-------- 4. dilation=2，pad=0，stride=1 --------
    {
        Tensor in(1.0,{1, 1, 5, 5});
        Tensor ker(1.0,{1, 1, 3, 3});
        Tensor expect({1, 1, 3, 3});
        expect(0,0,0,0)=4; expect(0,0,0,1)=6; expect(0,0,0,2)=4;
        expect(0,0,1,0)=6; expect(0,0,1,1)=9; expect(0,0,1,2)=6;
        expect(0,0,2,0)=4; expect(0,0,2,1)=6; expect(0,0,2,2)=4;
        Tensor out = conv2d(in, ker, 1, 1, 2);
        ok &= test2Tensor("dilate=2", out, expect);
    }
    return ok ? 0 : -1;
}


int matmul_test(){
    Tensor A(1.0,{2, 2, 3, 4});
    Tensor B(1.0,{2, 2, 4, 5});

    Tensor C = A.matmul(B);
    Tensor expect(4.0,{2,2,3,5});

    bool ok = test2Tensor("batchmatmul test",C,expect);
    return ok ? 0 : -1;
}

int maxpool2d_test(){
    return 0;
}


int transpose_test(){
    bool ok = true;
    {
        Tensor in({1, 1, 2, 2});
        in(0, 0, 0, 0) = 1; in(0, 0, 0, 1) = 2;
        in(0, 0, 1, 0) = 3; in(0, 0, 1, 1) = 4;

        Tensor expect({1, 1, 2, 2});
        expect(0, 0, 0, 0) = 1; expect(0, 0, 0, 1) = 3;
        expect(0, 0, 1, 0) = 2; expect(0, 0, 1, 1) = 4;

        ok &= test2Tensor("transpose test1",in.transpose(),expect);
    }
    {
        Tensor in({1, 1, 2, 3});
        in(0, 0, 0, 0) = 1; in(0, 0, 0, 1) = 2; in(0, 0, 0, 2) = 3;
        in(0, 0, 1, 0) = 4; in(0, 0, 1, 1) = 5; in(0, 0, 1, 2) = 6;

        Tensor expect({1, 1, 3, 2});
        expect(0, 0, 0, 0) = 1; expect(0, 0, 0, 1) = 4;
        expect(0, 0, 1, 0) = 2; expect(0, 0, 1, 1) = 5;
        expect(0, 0, 2, 0) = 3; expect(0, 0, 2, 1) = 6;

        ok &= test2Tensor("transpose test2",in.transpose(),expect);
    }

    {
        Tensor in({1, 1, 8, 3});
        in(0,0,0,0)= 1; in(0,0,0,1)= 2; in(0,0,0,2)= 3;
        in(0,0,1,0)= 4; in(0,0,1,1)= 5; in(0,0,1,2)= 6;
        in(0,0,2,0)= 7; in(0,0,2,1)= 8; in(0,0,2,2)= 9;
        in(0,0,3,0)=10; in(0,0,3,1)=11; in(0,0,3,2)=12;
        in(0,0,4,0)=13; in(0,0,4,1)=14; in(0,0,4,2)=15;
        in(0,0,5,0)=16; in(0,0,5,1)=17; in(0,0,5,2)=18;
        in(0,0,6,0)=19; in(0,0,6,1)=20; in(0,0,6,2)=21;
        in(0,0,7,0)=22; in(0,0,7,1)=23; in(0,0,7,2)=24;

        Tensor expect({1, 1, 3, 8});
        expect(0,0,0,0)= 1; expect(0,0,0,1)= 4; expect(0,0,0,2)= 7; expect(0,0,0,3)=10;
        expect(0,0,0,4)=13; expect(0,0,0,5)=16; expect(0,0,0,6)=19; expect(0,0,0,7)=22;
        expect(0,0,1,0)= 2; expect(0,0,1,1)= 5; expect(0,0,1,2)= 8; expect(0,0,1,3)=11;
        expect(0,0,1,4)=14; expect(0,0,1,5)=17; expect(0,0,1,6)=20; expect(0,0,1,7)=23;
        expect(0,0,2,0)= 3; expect(0,0,2,1)= 6; expect(0,0,2,2)= 9; expect(0,0,2,3)=12;
        expect(0,0,2,4)=15; expect(0,0,2,5)=18; expect(0,0,2,6)=21; expect(0,0,2,7)=24;
        ok &= test2Tensor("transpose test3",in.transpose(),expect);
    }
    return ok ? 0 : -1;
}

int tensor_test(){
    if(base_test() != 0){
        return -1;
    }
    if(ops_test() != 0){
        return -1;
    }
    if(conv2d_test() != 0){
        return -1;
    }
    if(matmul_test() != 0){
        return -1;
    }
    if(maxpool2d_test() != 0){
        return -1;
    }
    if(transpose_test() != 0){
        return -1;
    }
    return 0;
}