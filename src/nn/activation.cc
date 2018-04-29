#include "tensor.hh"
#include "activation.hh"

void nn::relu(const Tensor<float, 4> input, Tensor<float, 4> output) {

    for(nn::Index n = 0; n < input.dimension(0); n++){
        for(nn::Index c = 0; c < input.dimension(1); c++){
            for(nn::Index h = 0; h < input.dimension(2); h++){
                for(nn::Index w = 0; w < input.dimension(3); w++){
                    
                    float val = input(n, c, h, w);
                    val = (val > 0) ? val : 0;
                    output(n, c, h, w) = val;

                }
            }
        }
    }
    
}
