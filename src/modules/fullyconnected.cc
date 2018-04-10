#include "fullyconnected.hh"

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
