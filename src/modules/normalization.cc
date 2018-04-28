#include <cmath>

#include "normalization.hh"

#include "tensor.hh"
#include "blob1d.hh"
#include "blob4d.hh"

void NN::batch_norm(const Blob4D<float> &input, 
                    const Blob1D<float> &means,
                    const Blob1D<float> &variances,
                    const Blob1D<float> &weight,
                    const Blob1D<float> &bias,
                    Blob4D<float> &output,
                    const float eps) {

    for(size_t i = 0; i < input.batch_size; i++){
        for(size_t j = 0; j < input.channels; j++){

            const float channel_mean = means.get(j);
            const float channel_stddev = std::sqrt(variances.get(j) + eps);
            const float channel_weight = weight.get(j);
            const float channel_bias = bias.get(j);

            for(size_t k = 0; k < input.height; k++){
                for(size_t n = 0; n < input.width; n++){
                    
                    float val = input.get(i, j, k, n);
                    
                    val = (channel_weight * (val - channel_mean)) / channel_stddev + channel_bias;
                    output.set(val, i, j, k, n);
                    
                }
            }
        }
    }    
}

void NN::batch_norm(const Tensor<float, 4> input, 
                    const Tensor<float, 1> means,
                    const Tensor<float, 1> variances,
                    const Tensor<float, 1> weight,
                    const Tensor<float, 1> bias,
                    Tensor<float, 4> output,
                    const float eps) {

    for(NN::Index i = 0; i < input.dimension(0); i++){
        for(NN::Index j = 0; j < input.dimension(1); j++){

            const float channel_mean = means(j);
            const float channel_stddev = std::sqrt(variances(j) + eps);
            const float channel_weight = weight(j);
            const float channel_bias = bias(j);

            for(NN::Index k = 0; k < input.dimension(2); k++){
                for(NN::Index n = 0; n < input.dimension(3); n++){
                    
                    float val = input(i, j, k, n);
                    
                    val = (channel_weight * (val - channel_mean)) / channel_stddev + channel_bias;
                    output(i, j, k, n) = val;
                    
                }
            }
        }
    }    
}
