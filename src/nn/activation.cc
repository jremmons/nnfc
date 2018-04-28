#include "tensor.hh"
#include "activation.hh"

void NN::relu(const Tensor<float, 4> input, Tensor<float, 4> output) {

    for(NN::Index n = 0; n < input.dimension(0); n++){
        for(NN::Index c = 0; c < input.dimension(1); c++){
            for(NN::Index h = 0; h < input.dimension(2); h++){
                for(NN::Index w = 0; w < input.dimension(3); w++){
                    
                    float val = input(n, c, h, w);
                    val = (val > 0) ? val : 0;
                    output(n, c, h, w) = val;

                }
            }
        }
    }
    
}
