#include <cmath>

#include "normalization.hh"

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
