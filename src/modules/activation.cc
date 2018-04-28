#include "activation.hh"

#include "tensor.hh"
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
