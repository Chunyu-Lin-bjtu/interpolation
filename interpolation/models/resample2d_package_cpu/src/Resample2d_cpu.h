int Resample2d_cpu_forward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* output, int kernel_size);
int Resample2d_cpu_backward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* gradOutput, THFloatTensor* gradInput1, THFloatTensor* gradInput2, int kernel_size);
