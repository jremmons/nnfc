#include "tensor.hh"
#include "fullyconnected.hh"

void NN::fully_connected(const NN::Tensor<float, 4> input,
                         const NN::Tensor<float, 2> weights,
                         NN::Tensor<float, 4> output) {

    assert(input.dimension(0) == output.dimension(0));

    assert(input.dimension(2) == output.dimension(2));
    assert(input.dimension(2) == 1);

    assert(input.dimension(3) == output.dimension(3));
    assert(input.dimension(3) == 1);

    assert(input.dimension(1) == weights.dimension(1));
    assert(output.dimension(1) == weights.dimension(0));
    
    for(NN::Index i = 0; i < input.dimension(0); i++){
        for(NN::Index j = 0; j < output.dimension(1); j++){

            double val = 0;
            for(NN::Index k = 0; k < input.dimension(1); k++){
                
                float x = input(i, k, 0, 0);
                float w = weights(j, k);
                val += (w*x);
                   
            }

            output(i, j, 0, 0) = val;
            
        }
    }    
}
