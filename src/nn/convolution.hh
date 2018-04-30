#ifndef _NN_CONVOLUTION_H
#define _NN_CONVOLUTION_H

#include "tensor.hh"

namespace nn {

    void conv2d(const Tensor<float, 4> input,
                const Tensor<float, 4> kernel,
                Tensor<float, 4> output,
                const size_t stride,
                const size_t zero_padding);

}

#endif // _NN_CONVOLUTION_H
