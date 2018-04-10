
#include <cassert>

#include "pool.hh"
#include "blob4d.hh"

void NN::average_pooling(const Blob4D<float> &input,
                         Blob4D<float> &output) {

    assert(input.batch_size == output.batch_size);
    assert(input.channels == output.channels);
    
    for(size_t i = 0; i < input.batch_size; i++){
        for(size_t j = 0; j < input.channels; j++){
            
            double sum = 0.0;
            const size_t count = input.height * input.width;
            
            for(size_t k = 0; k < input.height; k++){
                for(size_t n = 0; n < input.width; n++){
                    
                    float val = input.get(i, j, k, n);
                    sum += static_cast<double>(val);
                }
            }
            
            const double average = sum / static_cast<double>(count);
            
            for(size_t k = 0; k < output.height; k++){
                for(size_t n = 0; n < output.width; n++){
                    
                    output.set(average, i, j, k, n);
                    
                }
            }
            
        }
    }
    
}
