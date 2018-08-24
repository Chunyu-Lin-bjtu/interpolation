#ifdef __cplusplus
    extern "C" {
#endif

void Resample2d_kernel_forward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* output, int kernel_size);

void Resample2d_kernel_backward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* gradOutput, THFloatTensor* gradInput1, THFloatTensor* gradInput2, int kernel_size);

#ifdef __cplusplus
    }
#endif
