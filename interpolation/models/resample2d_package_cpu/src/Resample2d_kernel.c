#include <TH.h>
#include <THGeneral.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define DIM0(TENSOR) (TENSOR[0])
#define DIM1(TENSOR) (TENSOR[1])
#define DIM2(TENSOR) (TENSOR[2])
#define DIM3(TENSOR) (TENSOR[3])

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[xx*yy*zz*ww+yy*zz*ww+zz*ww+ww])

#ifdef __cplusplus
    extern "C" {
#endif

int min(int a, int b) {
    return a<b ? a : b;
}

int max(int a, int b) {
    return a>b ? a : b;
}


void kernel_Resample2d_updateOutput(const int n, const float* input1, int* input1_size, int* input1_stride,const float* input2, int* input2_size, int* input2_stride, 
    float* output, int* output_size, int* output_stride, int kernel_size) {

     for(int index = 0;index < n;index++) {

            float val = 0.0;

	    int dim_b = DIM0(output_size);
	    int dim_c = DIM1(output_size);
	    int dim_h = DIM2(output_size);
	    int dim_w = DIM3(output_size);
	    int dim_chw = dim_c * dim_h * dim_w;
	    int dim_hw  = dim_h * dim_w;

	    int b = ( index / dim_chw ) % dim_b;
	    int c = ( index / dim_hw )  % dim_c;
	    int y = ( index / dim_w )   % dim_h;
	    int x = ( index          )  % dim_w;

	    float dx = DIM3_INDEX(input2, b, 0, y, x);
	    float dy = DIM3_INDEX(input2, b, 1, y, x);

	    float xf = (float)x + dx;
	    float yf = (float)y + dy;
	    float alpha = xf - floor(xf); // alpha
	    float beta = yf - floor(yf); // beta

	    int xL = max(min((int)floor(xf),    dim_w-1), 0);
	    int xR = max(min((int)(floor(xf)+1), dim_w -1), 0);
	    int yT = max(min((int)floor(yf),    dim_h-1), 0);
	    int yB = max(min((int)(floor(yf)+1),  dim_h-1), 0);

	    for (int fy = 0; fy < kernel_size; fy += 1) {
		for (int fx = 0; fx < kernel_size; fx += 1) {
		    val += (1. - alpha)*(1. - beta) * DIM3_INDEX(input1, b, c, yT + fy, xL + fx);
		    val +=    (alpha)*(1. - beta) * DIM3_INDEX(input1, b, c, yT + fy, xR + fx);
		    val +=    (1. - alpha)*(beta) * DIM3_INDEX(input1, b, c, yB + fy, xL + fx);
		    val +=       (alpha)*(beta) * DIM3_INDEX(input1, b, c, yB + fy, xR + fx);
		}
	    }

	    output[index] = val;   
  }

}


void kernel_Resample2d_backward_input1(
    const int n, const float* input1, int* input1_size, int* input1_stride, const float* input2, int* input2_size, int* input2_stride,
    const float* gradOutput, int*  gradOutput_size, int* gradOutput_stride, float* gradInput, int* gradInput_size, int* gradInput_stride, int kernel_size) {

    for(int index = 0;index < n;index++)
    {
            int dim_b = DIM0(gradOutput_size);
	    int dim_c = DIM1(gradOutput_size);
	    int dim_h = DIM2(gradOutput_size);
	    int dim_w = DIM3(gradOutput_size);
	    int dim_chw = dim_c * dim_h * dim_w;
	    int dim_hw  = dim_h * dim_w;

	    int b = ( index / dim_chw ) % dim_b;
	    int c = ( index / dim_hw )  % dim_c;
	    int y = ( index / dim_w )   % dim_h;
	    int x = ( index          )  % dim_w;

	    float dx = DIM3_INDEX(input2, b, 0, y, x);
	    float dy = DIM3_INDEX(input2, b, 1, y, x);

	    float xf = (float)x + dx;
	    float yf = (float)y + dy;
	    float alpha = xf - (int)xf; // alpha
	    float beta = yf - (int)yf; // beta

	    int idim_h = DIM2(input1_size);
	    int idim_w = DIM3(input1_size);

	    int xL = max(min( (int)floor(xf),    idim_w-1), 0);
	    int xR = max(min( (int)(floor(xf)+1), idim_w -1), 0);
	    int yT = max(min( (int)floor(yf),    idim_h-1), 0);
	    int yB = max(min( (int)(floor(yf)+1),  idim_h-1), 0);

	    for (int fy = 0; fy < kernel_size; fy += 1) {
		for (int fx = 0; fx < kernel_size; fx += 1) {
		    //atomicAdd(&DIM3_INDEX(gradInput, b, c, (yT + fy), (xL + fx)), (1-alpha)*(1-beta) * DIM3_INDEX(gradOutput, b, c, y, x));
		    //atomicAdd(&DIM3_INDEX(gradInput, b, c, (yT + fy), (xR + fx)),   (alpha)*(1-beta) * DIM3_INDEX(gradOutput, b, c, y, x));
		    //atomicAdd(&DIM3_INDEX(gradInput, b, c, (yB + fy), (xL + fx)),   (1-alpha)*(beta) * DIM3_INDEX(gradOutput, b, c, y, x));
		    //atomicAdd(&DIM3_INDEX(gradInput, b, c, (yB + fy), (xR + fx)),     (alpha)*(beta) * DIM3_INDEX(gradOutput, b, c, y, x));

		    DIM3_INDEX(gradInput, b, c, (yT + fy), (xL + fx)) = (1-alpha)*(1-beta) * DIM3_INDEX(gradOutput, b, c, y, x) + DIM3_INDEX(gradInput, b, c, (yT + fy), (xL + fx));
		    DIM3_INDEX(gradInput, b, c, (yT + fy), (xR + fx)) = (alpha)*(1-beta) * DIM3_INDEX(gradOutput, b, c, y, x) + DIM3_INDEX(gradInput, b, c, (yT + fy), (xR + fx));
		    DIM3_INDEX(gradInput, b, c, (yB + fy), (xL + fx)) = (1-alpha)*(beta) * DIM3_INDEX(gradOutput, b, c, y, x) + DIM3_INDEX(gradInput, b, c, (yB + fy), (xL + fx));
		    DIM3_INDEX(gradInput, b, c, (yB + fy), (xR + fx)) = (alpha)*(beta) * DIM3_INDEX(gradOutput, b, c, y, x) + DIM3_INDEX(gradInput, b, c, (yB + fy), (xR + fx));
		}
	    }
    }

}

void kernel_Resample2d_backward_input2(const int n, const float* input1, int* input1_size, int* input1_stride, const float* input2, int* input2_size, int* input2_stride,
    const float* gradOutput, int* gradOutput_size, int* gradOutput_stride, float* gradInput, int* gradInput_size, int* gradInput_stride, int kernel_size) {

    for(int index = 0;index < n;index++)
    {
	    float output = 0.0;
	    int kernel_rad = (kernel_size - 1)/2;

	    int dim_b = DIM0(gradInput_size);
	    int dim_c = DIM1(gradInput_size);
	    int dim_h = DIM2(gradInput_size);
	    int dim_w = DIM3(gradInput_size);
	    int dim_chw = dim_c * dim_h * dim_w;
	    int dim_hw  = dim_h * dim_w;

	    int b = ( index / dim_chw ) % dim_b;
	    int c = ( index / dim_hw )  % dim_c;
	    int y = ( index / dim_w )   % dim_h;
	    int x = ( index          )  % dim_w;

	    int odim_c = DIM1(gradOutput_size);

	    float dx = DIM3_INDEX(input2, b, 0, y, x);
	    float dy = DIM3_INDEX(input2, b, 1, y, x);

	    float xf = (float)x + dx;
	    float yf = (float)y + dy;

	    int xL = max(min( (int)floor(xf),    dim_w-1), 0);
	    int xR = max(min( (int)(floor(xf)+1), dim_w -1), 0);
	    int yT = max(min( (int)floor(yf),    dim_h-1), 0);
	    int yB = max(min( (int)(floor(yf)+1),  dim_h-1), 0);
	    
	    if (c % 2) {
		float gamma = 1 - (xf - floor(xf)); // alpha
		for (int i = 0; i <= 2*kernel_rad; ++i) {
		    for (int j = 0; j <= 2*kernel_rad; ++j) {
		        for (int ch = 0; ch < odim_c; ++ch) {
		            output += (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xL + i));
		            output -= (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xL + i));
		            output += (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xR + i));
		            output -= (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xR + i));
		        }
		    }
		}
	    }
	    else {
		float gamma = 1 - (yf - floor(yf)); // alpha
		for (int i = 0; i <= 2*kernel_rad; ++i) {
		    for (int j = 0; j <= 2*kernel_rad; ++j) {
		        for (int ch = 0; ch < odim_c; ++ch) {
		            output += (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xR + i));
		            output -= (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xL + i));
		            output += (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xR + i));
		            output -= (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xL + i));
		        }
		    }
		}

	    }

	    gradInput[index] = output;
    }

}

void Resample2d_kernel_forward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* output, int kernel_size) {
    int n = 0;


    int input1_size[4],input1_stride[4],input2_size[4],input2_stride[4];
    int output_size[4],output_stride[4];

    input1_size[0] = input1->size[0];
    input1_size[1] = input1->size[1];
    input1_size[2] = input1->size[2];
    input1_size[3] = input1->size[3];
    
    input1_stride[0] = input1->stride[0];
    input1_stride[1] = input1->stride[1];
    input1_stride[2] = input1->stride[2];
    input1_stride[3] = input1->stride[3];

    input2_size[0] = input2->size[0];
    input2_size[1] = input2->size[1];
    input2_size[2] = input2->size[2];
    input2_size[3] = input2->size[3];
    
    input2_stride[0] = input2->stride[0];
    input2_stride[1] = input2->stride[1];
    input2_stride[2] = input2->stride[2];
    input2_stride[3] = input2->stride[3];

    output_size[0] = output->size[0];
    output_size[1] = output->size[1];
    output_size[2] = output->size[2];
    output_size[3] = output->size[3];

    output_stride[0] = output->stride[0];
    output_stride[1] = output->stride[1];
    output_stride[2] = output->stride[2];
    output_stride[3] = output->stride[3];
    

    //const long4 input1_size = make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]);
    //const long4 input1_stride = make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]);

    //const long4 input2_size = make_long4(input2->size[0], input2->size[1], input2->size[2], input2->size[3]);
    //const long4 input2_stride = make_long4(input2->stride[0], input2->stride[1], input2->stride[2], input2->stride[3]);

    //const long4 output_size = make_long4(output->size[0], output->size[1], output->size[2], output->size[3]);
    //const long4 output_stride = make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3]);

    n = THFloatTensor_nElement(output);
    kernel_Resample2d_updateOutput(n, THFloatTensor_data(input1), input1_size, input1_stride, THFloatTensor_data(input2), input2_size, input2_stride,
        THFloatTensor_data(output), output_size, output_stride, kernel_size);

}

void Resample2d_kernel_backward(THFloatTensor* input1, THFloatTensor* input2, THFloatTensor* gradOutput, THFloatTensor* gradInput1, THFloatTensor* gradInput2, int kernel_size) {
    int n = 0;

    int input1_size[4],input1_stride[4],input2_size[4],input2_stride[4];
    int gradOutput_size[4],gradOutput_stride[4],gradInput1_size[4],gradInput1_stride[4];
    int gradInput2_size[4],gradInput2_stride[4];

    input1_size[0] = input1->size[0];
    input1_size[1] = input1->size[1];
    input1_size[2] = input1->size[2];
    input1_size[3] = input1->size[3];
    
    input1_stride[0] = input1->stride[0];
    input1_stride[1] = input1->stride[1];
    input1_stride[2] = input1->stride[2];
    input1_stride[3] = input1->stride[3];

    input2_size[0] = input2->size[0];
    input2_size[1] = input2->size[1];
    input2_size[2] = input2->size[2];
    input2_size[3] = input2->size[3];
    
    input2_stride[0] = input2->stride[0];
    input2_stride[1] = input2->stride[1];
    input2_stride[2] = input2->stride[2];
    input2_stride[3] = input2->stride[3];
    
    gradOutput_size[0] = gradOutput->size[0];
    gradOutput_size[1] = gradOutput->size[1];
    gradOutput_size[2] = gradOutput->size[2];
    gradOutput_size[3] = gradOutput->size[3];
   
    gradOutput_stride[0]=gradOutput->stride[0];
    gradOutput_stride[1]=gradOutput->stride[1];
    gradOutput_stride[2]=gradOutput->stride[2];
    gradOutput_stride[3]=gradOutput->stride[3];

    gradInput1_size[0] = gradInput1->size[0];
    gradInput1_size[1] = gradInput1->size[1];
    gradInput1_size[2] = gradInput1->size[2];
    gradInput1_size[3] = gradInput1->size[3];

    gradInput1_stride[0] = gradInput1->stride[0];
    gradInput1_stride[1] = gradInput1->stride[1];
    gradInput1_stride[2] = gradInput1->stride[2];
    gradInput1_stride[3] = gradInput1->stride[3];


    gradInput2_size[0] = gradInput2->size[0];
    gradInput2_size[1] = gradInput2->size[1];
    gradInput2_size[2] = gradInput2->size[2];
    gradInput2_size[3] = gradInput2->size[3];
    
    gradInput2_stride[0] = gradInput2->stride[0];
    gradInput2_stride[1] = gradInput2->stride[1];
    gradInput2_stride[2] = gradInput2->stride[2];
    gradInput2_stride[3] = gradInput2->stride[3];
    
    
    //const long4 input1_size = make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]);
    //const long4 input1_stride = make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]);

    //const long4 input2_size = make_long4(input2->size[0], input2->size[1], input2->size[2], input2->size[3]);
    //const long4 input2_stride = make_long4(input2->stride[0], input2->stride[1], input2->stride[2], input2->stride[3]);

    //const long4 gradOutput_size = make_long4(gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
    //const long4 gradOutput_stride = make_long4(gradOutput->stride[0], gradOutput->stride[1], gradOutput->stride[2], gradOutput->stride[3]);

    //const long4 gradInput1_size = make_long4(gradInput1->size[0], gradInput1->size[1], gradInput1->size[2], gradInput1->size[3]);
    //const long4 gradInput1_stride = make_long4(gradInput1->stride[0], gradInput1->stride[1], gradInput1->stride[2], gradInput1->stride[3]);

    n = THFloatTensor_nElement(gradOutput);
    kernel_Resample2d_backward_input1(n, THFloatTensor_data(input1), input1_size, input1_stride, THFloatTensor_data(input2), input2_size, input2_stride,
        THFloatTensor_data(gradOutput), gradOutput_size, gradOutput_stride, THFloatTensor_data(gradInput1), gradInput1_size, gradInput1_stride, kernel_size
    );

    //const long4 gradInput2_size = make_long4(gradInput2->size[0], gradInput2->size[1], gradInput2->size[2], gradInput2->size[3]);
    //const long4 gradInput2_stride = make_long4(gradInput2->stride[0], gradInput2->stride[1], gradInput2->stride[2], gradInput2->stride[3]);

    n = THFloatTensor_nElement(gradInput2);
    kernel_Resample2d_backward_input2(n, THFloatTensor_data(input1), input1_size, input1_stride, THFloatTensor_data(input2), input2_size, input2_stride,
        THFloatTensor_data(gradOutput), gradOutput_size, gradOutput_stride, THFloatTensor_data(gradInput2), gradInput2_size, gradInput2_stride, kernel_size
    );

}


#ifdef __cplusplus
    }
#endif
