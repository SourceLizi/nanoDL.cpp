#include "../src/autograd.hpp"
#include <iostream>
#define M_PI 3.14159265358979323846

#define FP_COMPARE_EPS 1e-5

double finite_diff(AutoGradGraph<double>& g,Value<double>& var,Value<double>& out){
    double eps = 1e-6;
    double orig = var.value();

    var.value() = orig + eps;
    g.forward();
    double Lp = out.value();
    var.value() = orig - eps;
    g.forward();
    double Lm = out.value();
    var.value() = orig;       
    return (Lp - Lm) / (2*eps);
};

bool single_fun_grad_test(AutoGradGraph<double>& g,Value<double>& x,Value<double>& y){
    g.build(y);
    g.forward();
    g.backward(1.0);

    double dx = finite_diff(g,x,y);
    if(abs(dx-x.grad()) < FP_COMPARE_EPS){
        return true;
    }else{
        std::cout << "Gradients:\n";
        std::cout << "finite_diff : dy/dx = " << dx << "\n";
        std::cout << "graph grad  : dy/dx = " << x.grad() << "\n";
        return false;
    }
}


int sincos_scalar_test(){
    for(int i = 0;i < 20;i++){
        AutoGradGraph<double> g;
        Value<double> x = g.var((double)i/10.0*M_PI);
        Value<double> y = g.sin(x);

        if(!single_fun_grad_test(g,x,y)){
            std::cout << "Check dsin(x)/dx failed at x=" << x.value() << std::endl;
            return -1;
        }
    }
    for(int i = 0;i < 20;i++){
        AutoGradGraph<double> g;
        Value<double> x = g.var((double)i/10.0*M_PI);
        Value<double> y = g.cos(x);

        if(!single_fun_grad_test(g,x,y)){
            std::cout << "Check dcos(x)/dx failed at x="  << x.value() << std::endl;
            return -1;
        }
    }
    return 0;
}

int pow_scalar_test(double n){
    AutoGradGraph<double> g;
    Value<double> x = g.var(23.7);
    Value<double> y = g.pow(x,n);

    if(!single_fun_grad_test(g,x,y)){
        std::cout << "Check dpow(x,n)/dx failed at n=" << n << std::endl;
        return -1;
    }
    return 0;
}

int explog_scalar_test(){
    AutoGradGraph<double> g1;
    Value<double> x = g1.var(2.6);
    Value<double> y = g1.exp(x);

    if(!single_fun_grad_test(g1,x,y)){
        std::cout << "Check dexp(x)/dx failed" << std::endl;
        return -1;
    }

    AutoGradGraph<double> g2;
    Value<double> x2 = g2.var(1.6);
    Value<double> y2 = g2.log(x2);

    if(!single_fun_grad_test(g2,x2,y2)){
        std::cout << "Check dlog(x)/dx failed" << std::endl;
        return -1;
    }
    return 0;
}


int multivar_scalar_test(){
    AutoGradGraph<double> g;
    // 定义变量
    Value<double> x = g.var(1.5);
    Value<double> y = g.var(-0.5);
    Value<double> z = g.var(2.0);

    Value<double> L = (x * y + g.pow(y, 2.0))/z + g.sin(x) + g.cos(z) - x;
    // 编译计算图
    g.build(L);
    g.forward();
    g.backward(1.0);

    std::cout << "Forward: L = " << L.value() << "\n";
    std::cout << "Gradients:\n";
    std::cout << "  dL/dx = " << x.grad() << "\n";
    std::cout << "  dL/dy = " << y.grad() << "\n";
    std::cout << "  dL/dz = " << z.grad() << "\n";

    double nx = finite_diff(g,x,L);
    double ny = finite_diff(g,y,L);
    double nz = finite_diff(g,z,L);

    std::cout << "\nFinite-diff check:\n";
    std::cout << "  dL/dx (num) = " << nx << "\n";
    std::cout << "  dL/dy (num) = " << ny << "\n";
    std::cout << "  dL/dz (num) = " << nz << "\n";
    if((abs(nx-x.grad()) < FP_COMPARE_EPS) && (abs(ny-y.grad()) < FP_COMPARE_EPS)
        && (abs(nz-z.grad()) < FP_COMPARE_EPS)){
        return 0;
    }else{
        std::cout << "Check multivar grad failed" << std::endl;
        return -1;
    }
}

int multivar_scalar_test2(){
    AutoGradGraph<double> g;
    // 定义变量
    Value<double> x = g.var(1.1);
    Value<double> y = g.var(8.5);
    Value<double> z = g.var(2.1);

    Value<double> L = g.log(x + y)*z - x*y/(x+y) +g.exp(x)/z;
    // 编译计算图
    g.build(L);
    g.forward();
    g.backward(1.0);

    std::cout << "Forward: L = " << L.value() << "\n";
    std::cout << "Gradients:\n";
    std::cout << "  dL/dx = " << x.grad() << "\n";
    std::cout << "  dL/dy = " << y.grad() << "\n";
    std::cout << "  dL/dz = " << z.grad() << "\n";

    double nx = finite_diff(g,x,L);
    double ny = finite_diff(g,y,L);
    double nz = finite_diff(g,z,L);

    std::cout << "\nFinite-diff check:\n";
    std::cout << "  dL/dx (num) = " << nx << "\n";
    std::cout << "  dL/dy (num) = " << ny << "\n";
    std::cout << "  dL/dz (num) = " << nz << "\n";
    if((abs(nx-x.grad()) < FP_COMPARE_EPS) && (abs(ny-y.grad()) < FP_COMPARE_EPS)
        && (abs(nz-z.grad()) < FP_COMPARE_EPS)){
        return 0;
    }else{
        std::cout << "Check multivar grad failed" << std::endl;
        return -1;
    }
}

int exception_check(){
    AutoGradGraph<double> g1;
    Value<double> x = g1.var(2.6);
    AutoGradGraph<double> g2;
    Value<double> y = g2.var(1.2);
    try{
        Value<double> z = x + y;
        return -1;
    }
    catch(const std::exception& e){
        std::cout << "Exception trigger success" << std::endl;
    }
    try{
        Value<double> l = g2.exp(x);
        return -1;
    }
    catch(const std::exception& e){
        std::cout << "Exception trigger success" << std::endl;
    }
    return 0;
}

int autograd_scalar_test(){
    if(sincos_scalar_test() != 0){
        return -1;
    }
    if(explog_scalar_test() != 0){
        return -1;
    }
    if(pow_scalar_test(2.0) != 0){
        return -1;
    }
    if(pow_scalar_test(3.0) != 0){
        return -1;
    }

    if(multivar_scalar_test() != 0){
        return -1;
    }
    if(multivar_scalar_test2() != 0){
        return -1;
    }

    if(exception_check() != 0){
        return -1;
    }
    return 0;
}