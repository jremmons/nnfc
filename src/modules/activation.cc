#include "activation.hh"

#include "blob4d.hh"

void NN::relu(const Blob4D<float> &input, Blob4D<float> &output) {

    for(size_t i = 0; i < input.batch_size; i++){
        for(size_t j = 0; j < input.channels; j++){
            for(size_t k = 0; k < input.height; k++){
                for(size_t n = 0; n < input.width; n++){

                    float val = input.get(i, j, k, n);
                    val = (val > 0) ? val : 0;
                    output.set(val, i, j, k, n);
                    
                }
            }
        }
    }
    
}
