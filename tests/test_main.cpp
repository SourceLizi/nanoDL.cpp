#include <stdio.h>
#include <stdlib.h>

#define NUM_FUNC 3

int autograd_scalar_test();
int tensor_test();
int autograd_tensor_test();

int (*fun_ptr[NUM_FUNC])()={tensor_test, autograd_scalar_test, autograd_tensor_test};

int main(int argc, char **argv){
    if (argc < 2) {
        return -1;
    }
    int type = atoi(argv[1]);
    if(type >= 0 && type < NUM_FUNC){
        return fun_ptr[type]();
    }else{
        return -1;
    }
}