#include "convolution.hh"
#include "tensor.hh"

#include <iostream>

void NN::conv2d(const Tensor<float, 4> input,
                const Tensor<float, 4> kernel,
                Tensor<float, 4> output,
                const size_t stride,
                const size_t zero_padding) {

    
    assert(input.dimension(0) == output.dimension(0));
    assert(output.dimension(1) == kernel.dimension(0));
    assert(input.dimension(1) == kernel.dimension(1)); 
   
    const NN::Index h_extent = (input.dimension(2) - kernel.dimension(2) + 2*zero_padding) / stride + 1;
    const NN::Index w_extent = (input.dimension(3) - kernel.dimension(3) + 2*zero_padding) / stride + 1;

    assert(output.dimension(2) == h_extent);
    assert(output.dimension(3) == w_extent);

    for(NN::Index i = 0; i < input.dimension(0); i++){

        for(NN::Index j = 0; j < output.dimension(1); j++){

            for(NN::Index n = 0; n < output.dimension(2); n++){
                for(NN::Index m = 0; m < output.dimension(3); m++){

                    float val = 0.0;

                    const int64_t y = static_cast<int64_t>(stride * n) - zero_padding;
                    const int64_t x = static_cast<int64_t>(stride * m) - zero_padding;

                    for(NN::Index k = 0; k < input.dimension(1); k++){
                        for(NN::Index h = 0; h < kernel.dimension(2); h++){ 
                            for(NN::Index w = 0; w < kernel.dimension(3); w++){
                                
                                const int64_t y_image = y + h;
                                const int64_t x_image = x + w;

                                if(0 <= y_image and y_image < static_cast<int64_t>(input.dimension(2)) and \
                                   0 <= x_image and x_image < static_cast<int64_t>(input.dimension(3))){

                                    float kernel_weight = kernel(j, k, h, w);
                                    float inp = input(i, k, y_image, x_image);
                                    val += kernel_weight * inp;
                                }
                            }
                        }
                    }
                    
                    output(i, j, n, m) = val;
                    
                }
            }            
        }
    }
}
