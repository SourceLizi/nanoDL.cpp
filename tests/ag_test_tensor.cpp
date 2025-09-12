#include "../src/autograd.hpp"
#include "../src/Tensor.hpp"
#include <iostream>
#define M_PI 3.14159265358979323846

#define FP_COMPARE_EPS 1e-5
#define DIFF_EPS 1e-6

void print_shape(const std::vector<size_t>& shape);
bool test2Tensor(const std::string& name,
              const Tensor& out,
              const Tensor& expected);

bool fun_grad_test(AutoGradGraph<Tensor>& g,Value<Tensor>& x,Value<Tensor>& y){
    g.build(y);
    g.forward();
    g.backward(Tensor(1.0,y.value().shape()));

    
    Tensor eps = Tensor(DIFF_EPS,x.value().shape());
    Tensor orig = x.value();

    x.value() = orig + eps;
    g.forward();
    Tensor Lp = y.value();
    x.value() = orig - eps;
    g.forward();
    Tensor Lm = y.value();
    x.value() = orig;       
    Tensor dx2 =  (Lp - Lm);
    Tensor dx_graph = x.grad();

    for(int i = 0;i < dx2.size();i++){
        if(abs(dx2[i]/DIFF_EPS/2 - dx_graph[i]) > FP_COMPARE_EPS){
            std::cout << "Gradients check failed\n";
            return false;
        }
    }
    return true;
}

int conv2d_grad_test(){
    AutoGradGraph<Tensor> g;
    Value<Tensor> x = g.var(Tensor(1.0, {1, 1, 3, 4}));
    Value<Tensor> ker = g.var(Tensor(1.0, {1, 1, 3, 3}));
    Value<Tensor> y = g.conv2d(x,ker,1,1,1);

    g.build(y);
    g.forward();
    g.backward(Tensor(1.0,y.value().shape()));

    Tensor expect = Tensor({1,1,3,4});
    expect(0, 0, 0, 0) = 4; expect(0, 0, 0, 1) = 6; expect(0, 0, 0, 2) = 6; expect(0, 0, 0, 3) = 4;
    expect(0, 0, 1, 0) = 6; expect(0, 0, 1, 1) = 9; expect(0, 0, 1, 2) = 9; expect(0, 0, 1, 3) = 6;
    expect(0, 0, 2, 0) = 4; expect(0, 0, 2, 1) = 6; expect(0, 0, 2, 2) = 6; expect(0, 0, 2, 3) = 4;

    if(!test2Tensor("conv2d x grad",x.grad(),expect)){
        std::cout << "Check matmul conv2d x grad failed" << std::endl;
        return -1;
    }
    expect = Tensor({1,1,3,3});
    expect(0, 0, 0, 0) = 6; expect(0, 0, 0, 1) = 8; expect(0, 0, 0, 2) = 6;
    expect(0, 0, 1, 0) = 9; expect(0, 0, 1, 1) = 12; expect(0, 0, 1, 2) = 9;
    expect(0, 0, 2, 0) = 6; expect(0, 0, 2, 1) = 8; expect(0, 0, 2, 2) = 6;
    if(!test2Tensor("conv2d kernel grad",ker.grad(),expect)){
        std::cout << "Check matmul conv2d kernel grad failed" << std::endl;
        return -1;
    }
    return 0;
}

int matmul_grad_test(){
    /* Pytorch validate
    A=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]],requires_grad=True)
    B=torch.tensor([[1.0,7.0],[9.0,3.0],[2.0,6.0]],requires_grad=True)
    C = A @ B
    C.backward(torch.ones_like(C))
    print(A.grad, B.grad)
    */
    AutoGradGraph<Tensor> g;
    Value<Tensor> A = g.var(Tensor({1, 1, 2, 3}));
    A.value()(0, 0, 0, 0) = 1; A.value()(0, 0, 0, 1) = 2; A.value()(0, 0, 0, 2) = 3;
    A.value()(0, 0, 1, 0) = 4; A.value()(0, 0, 1, 1) = 5; A.value()(0, 0, 1, 2) = 6;

    Value<Tensor> B = g.var(Tensor({1, 1, 3, 2}));
    B.value()(0, 0, 0, 0) = 1; B.value()(0, 0, 0, 1) = 7;
    B.value()(0, 0, 1, 0) = 9; B.value()(0, 0, 1, 1) = 3;
    B.value()(0, 0, 2, 0) = 2; B.value()(0, 0, 2, 1) = 6;

    Value<Tensor> C = g.matmul(A,B);

    g.build(C);
    g.forward();
    g.backward(Tensor(1.0,C.value().shape()));

    Tensor expect = Tensor({1,1,2,3});
    expect(0, 0, 0, 0) = 8; expect(0, 0, 0, 1) = 12; expect(0, 0, 0, 2) = 8;
    expect(0, 0, 1, 0) = 8; expect(0, 0, 1, 1) = 12; expect(0, 0, 1, 2) = 8;

    if(!test2Tensor("matmul A grad",A.grad(),expect)){
        std::cout << "Check matmul A grad failed" << std::endl;
        return -1;
    }
    expect = Tensor({1,1,3,2});
    expect(0, 0, 0, 0) = 5; expect(0, 0, 0, 1) = 5;
    expect(0, 0, 1, 0) = 7; expect(0, 0, 1, 1) = 7;
    expect(0, 0, 2, 0) = 9; expect(0, 0, 2, 1) = 9;
    if(!test2Tensor("matmul B grad",B.grad(),expect)){
        std::cout << "Check matmul B grad failed" << std::endl;
        return -1;
    }
    return 0;
}

int elemwise_fcn_test(){
    {
        AutoGradGraph<Tensor> g;
        Value<Tensor> x = g.var(Tensor(23.7,{2,2}));
        Value<Tensor> y = g.pow(x,2);

        if(!fun_grad_test(g,x,y)){
            std::cout << "Check dpow(x,n)/dx failed at n=" << 2 << std::endl;
            return -1;
        }
    }
    for(int i = 0;i < 20;i++){
        AutoGradGraph<Tensor> g;
        Value<Tensor> x = g.var(Tensor((double)i/10.0*M_PI,{2,3}));
        Value<Tensor> y = g.sin(x);

        if(!fun_grad_test(g,x,y)){
            std::cout << "Check dsin(x)/dx failed" << std::endl;
            return -1;
        }
    }
    for(int i = 0;i < 20;i++){
        AutoGradGraph<Tensor> g;
        Value<Tensor> x = g.var(Tensor((double)i/10.0*M_PI,{5,2}));
        Value<Tensor> y = g.cos(x);

        if(!fun_grad_test(g,x,y)){
            std::cout << "Check dcos(x)/dx failed at x" << std::endl;
            return -1;
        }
    }
    
    
    return 0;
}

int autograd_tensor_test(){
    if(elemwise_fcn_test() != 0){
        return -1;
    }
    if(matmul_grad_test() != 0){
        return -1;
    }
    if(conv2d_grad_test() != 0){
        return -1;
    }
    return 0;
}