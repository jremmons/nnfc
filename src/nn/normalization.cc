#include "tensor.hh"
#include "normalization.hh"

#include <cmath>

void nn::batch_norm(const Tensor<float, 4> input, 
                    const Tensor<float, 1> means,
                    const Tensor<float, 1> variances,
                    const Tensor<float, 1> weight,
                    const Tensor<float, 1> bias,
                    Tensor<float, 4> output,
                    const float eps) {

    for(nn::Index i = 0; i < input.dimension(0); i++){
        for(nn::Index j = 0; j < input.dimension(1); j++){

            const float channel_mean = means(j);
            const float channel_stddev = std::sqrt(variances(j) + eps);
            const float channel_weight = weight(j);
            const float channel_bias = bias(j);

            for(nn::Index k = 0; k < input.dimension(2); k++){
                for(nn::Index n = 0; n < input.dimension(3); n++){
                    
                    float val = input(i, j, k, n);
                    
                    val = (channel_weight * (val - channel_mean)) / channel_stddev + channel_bias;
                    output(i, j, k, n) = val;
                    
                }
            }
        }
    }    
}
