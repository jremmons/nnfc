#include "tensor.hh"
#include "fullyconnected.hh"

void nn::fully_connected(const nn::Tensor<float, 4> input,
                         const nn::Tensor<float, 2> weights,
                         nn::Tensor<float, 4> output) {

    assert(input.dimension(0) == output.dimension(0));

    assert(input.dimension(2) == output.dimension(2));
    assert(input.dimension(2) == 1);

    assert(input.dimension(3) == output.dimension(3));
    assert(input.dimension(3) == 1);

    assert(input.dimension(1) == weights.dimension(1));
    assert(output.dimension(1) == weights.dimension(0));
    
    for(nn::Index i = 0; i < input.dimension(0); i++){
        for(nn::Index j = 0; j < output.dimension(1); j++){

            double val = 0;
            for(nn::Index k = 0; k < input.dimension(1); k++){
                
                float x = input(i, k, 0, 0);
                float w = weights(j, k);
                val += (w*x);
                   
            }

            output(i, j, 0, 0) = val;
            
        }
    }    
}

void nn::fully_connected_with_bias(const nn::Tensor<float, 4> input,
                                   const nn::Tensor<float, 2> weights,
                                   const nn::Tensor<float, 1> bias,
                                   nn::Tensor<float, 4> output) {
    
    assert(input.dimension(0) == output.dimension(0));

    assert(input.dimension(2) == output.dimension(2));
    assert(input.dimension(2) == 1);

    assert(input.dimension(3) == output.dimension(3));
    assert(input.dimension(3) == 1);

    assert(input.dimension(1) == weights.dimension(1));
    assert(output.dimension(1) == weights.dimension(0));
    
    for(nn::Index i = 0; i < input.dimension(0); i++){
        for(nn::Index j = 0; j < output.dimension(1); j++){

            double val = 0;
            for(nn::Index k = 0; k < input.dimension(1); k++){
                
                float x = input(i, k, 0, 0);
                float w = weights(j, k);
                val += (w*x);
                   
            }

            output(i, j, 0, 0) = val + bias(j);
            
        }
    }    
}
