#include <TH/TH.h>
#include <time.h>

// https://github.com/torch/TH/blob/master/generic/THTensor.h

int add_forward(THFloatTensor *input1, THFloatTensor *input2,
		       THFloatTensor *output)
{

    /* if (!THFloatTensor_isSameSizeAs(input1, input2)) */
    /*     return 0; */
    /* float* a = THFloatTensor_data(input1); */
    /* //printf("%f\n", a[10]); */

    /* int dims = THFloatTensor_nDimension(input1); */
    /* printf("dims: %i\n", dims); */

    /* long storage_offset = THFloatTensor_storageOffset(input1);  */
    /* printf("storage_offset: %i\n", storage_offset); */
   
    /* int i; */
    /* printf("shape: "); */
    /* for(i = 0; i < dims; i++){ */

    /*     long size = THFloatTensor_size(input1, i); */
    /*     printf("%i ", size); */

    /* } */
    /* printf("\n"); */

    /* printf("stride: "); */
    /* for(i = 0; i < dims; i++){ */

    /*     long stride = THFloatTensor_stride(input1, i); */
    /*     printf("%i ", stride); */

    /* } */
    /* printf("\n");     */


    int i;
    for(i = 0; i < input1->nDimension; i++){
        printf("%d ", input1->size[i]);
    }
    printf("\n");
    
    /* THFloatTensor_resizeAs(output, input1); */
    /* THFloatTensor_cadd(output, input1, 1.0, input2); */

  return 1;
}

int add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  /* THFloatTensor_resizeAs(grad_input, grad_output); */
  /* THFloatTensor_fill(grad_input, 1); */
  return 1;
}
