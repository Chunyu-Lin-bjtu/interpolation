#include <TH.h>
#include <THGeneral.h>
#include "Resample2d_kernel.h"

int Resample2d_cpu_forward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* output, int kernel_size) {
    Resample2d_kernel_forward(input1, input2, output, kernel_size);
    return 1;
}


int Resample2d_cpu_backward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* gradOutput, THFloatTensor* gradInput1, THFloatTensor* gradInput2, int kernel_size) {
    Resample2d_kernel_backward(input1, input2, gradOutput, gradInput1, gradInput2, kernel_size);

    return 1;
}
