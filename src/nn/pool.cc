#include "tensor.hh"
#include "pool.hh"

#include <cassert>
#include <iostream>

void nn::average_pooling(const Tensor<float, 4> input,
                         Tensor<float, 4> output) {

    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));

    for(nn::Index n = 0; n < input.dimension(0); n++){
        std::cerr << "n:" << n << std::endl;
        for(nn::Index c = 0; c < input.dimension(1); c++){
            
            float sum = 0.0;
            const nn::Index count = input.dimension(2) * input.dimension(3);
            
            for(nn::Index h = 0; h < input.dimension(2); h++){
                for(nn::Index w = 0; w < input.dimension(3); w++){

                    float val = input(n, c, h, w);
                    sum += val;
                    
                }
                std::cerr << std::endl;
            }
            
            const float average = sum / count;

            
            for(nn::Index h = 0; h < output.dimension(2); h++){
                for(nn::Index w = 0; w < output.dimension(3); w++){
                    
                    output(n, c, h, w) = average;
                    
                }
            }
            
        }
    }    
}

