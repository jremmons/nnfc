#include "convolution.hh"

#include "tensor.hh"
#include "blob3d.hh"
#include "blob4d.hh"

#include <iostream>

void NN::conv2d(const Blob4D<float> &input,
                const Blob4D<float> &kernel,
                Blob4D<float> &output,
                const size_t stride,
                const size_t zero_padding) {

    
    assert(input.batch_size == output.batch_size);
    assert(output.channels == kernel.out_channels);
    assert(input.channels == kernel.in_channels); 
   
    const size_t h_extent = (input.height - kernel.height + 2*zero_padding) / stride + 1;
    const size_t w_extent = (input.width - kernel.width + 2*zero_padding) / stride + 1;

    assert(output.height == h_extent);
    assert(output.width == w_extent);

    for(size_t i = 0; i < input.batch_size; i++){

        for(size_t j = 0; j < output.channels; j++){

            for(size_t n = 0; n < output.height; n++){
                for(size_t m = 0; m < output.width; m++){

                    float val = 0.0;

                    const int64_t y = static_cast<int64_t>(stride * n) - zero_padding;
                    const int64_t x = static_cast<int64_t>(stride * m) - zero_padding;

                    for(size_t k = 0; k < input.channels; k++){
                        for(size_t h = 0; h < kernel.height; h++){ 
                            for(size_t w = 0; w < kernel.width; w++){
                                
                                const int64_t y_image = y + h;
                                const int64_t x_image = x + w;

                                if(0 <= y_image and y_image < static_cast<int64_t>(input.height) and \
                                   0 <= x_image and x_image < static_cast<int64_t>(input.width)){

                                    float kernel_weight = kernel.get(j, k, h, w);
                                    float inp = input.get(i, k, y_image, x_image);
                                    val += kernel_weight * inp;
                                }
                            }
                        }
                    }
                    
                    output.set(val, i, j, n, m);
                    
                }
            }            
        }
    }
}

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
