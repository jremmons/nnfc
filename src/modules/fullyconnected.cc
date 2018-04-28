#include "fullyconnected.hh"

#include "tensor.hh"
#include "blob2d.hh"
#include "blob4d.hh"

void NN::fully_connected(const Blob4D<float> &input,
                         const Blob2D<float> &weights,
                         Blob4D<float> &output) {

    assert(input.batch_size == output.batch_size);

    assert(input.height == output.height);
    assert(input.height == 1);

    assert(input.width == output.width);
    assert(input.width == 1);

    assert(input.channels == weights.in_channels);
    assert(output.channels == weights.out_channels);
    
    for(size_t i = 0; i < input.batch_size; i++){
        for(size_t j = 0; j < output.channels; j++){

            double val = 0;
            for(size_t k = 0; k < input.channels; k++){
                
                float x = input.get(i, k, 0, 0);
                float w = weights.get(j, k);
                val += (w*x);
                   
            }

            output.set(val, i, j, 0, 0);
            
        }
    }
    
}

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
