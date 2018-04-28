#ifndef _NN_CONVOLUTION
#define _NN_CONVOLUTION

#include "tensor.hh"

namespace NN {

    void conv2d(const Tensor<float, 4> input,
                const Tensor<float, 4> kernel,
                Tensor<float, 4> output,
                const size_t stride,
                const size_t zero_padding);
    
}

#endif // _NN_CONVOLUTION
