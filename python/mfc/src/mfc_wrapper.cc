#include <TH/TH.h>
#include <stdio.h>

#include <iostream>

#include "noop.hh"

// functions must 'extern "C"' in order to be callable from within pytorch/python
// https://github.com/torch/TH/blob/master/generic/THTensor.h

extern "C" int add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output)
{
    Noop n;

    std::cout << n.encode(1000) << "\n";
    
    int i;
    for(i = 0; i < input1->nDimension; i++){
        printf("%d ", input1->size[i]);
    }
    printf("\n");

  return 1;
}

extern "C" int add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  /* THFloatTensor_resizeAs(grad_input, grad_output); */
  /* THFloatTensor_fill(grad_input, 1); */
  return 1;
}
