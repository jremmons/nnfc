#include "convolution.hh"

#include "blob3d.hh"
#include "blob4d.hh"

#include <iostream>

void NN::conv2d(const Blob4D<float> &input,
                const Blob4D<float> &kernel,
                Blob4D<float> &output,
                const size_t stride,
                const size_t zero_padding) {


    std::cerr << "hello world" << std::endl;
    
    assert(input.batch_size == output.batch_size);
    assert(output.channels == kernel.out_channels);
    assert(input.channels == kernel.in_channels); 
   
    const size_t h_extent = (input.height - kernel.height + 2*zero_padding) / stride + 1;
    // const size_t w_extent = (input.width - kernel.width + 2*zero_padding) / stride + 1;

    assert(output.height == h_extent);
    // assert(output.width == w_extent);
    
    std::cerr << "hello world" << std::endl;

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

                                    std::cerr << j << " " << k << " " << h << " " << w << std::endl;
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
