
#include <cassert>
#include <iostream>

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

// void NN::average_pooling(const Tensor<float, 4> input,
//                          Tensor<float, 4> output) {

//     // assert(input.batch_size == output.batch_size);
//     // assert(input.channels == output.channels);
    
//     for(NN::Index i = 0; i < input.dimension(0); i++){
//         for(NN::Index j = 0; j < input.dimension(1); j++){
            
//             double sum = 0.0;
//             const NN::Index count = input.dimension(2) * input.dimension(3);
            
//             for(NN::Index k = 0; k < input.dimension(2); k++){
//                 for(NN::Index n = 0; n < input.dimension(3); n++){
                    
//                     float val = input(i, j, k, n);
//                     sum += static_cast<double>(val);
//                 }
//             }
            
//             const double average = sum / static_cast<double>(count);
            
//             for(NN::Index k = 0; k < output.dimension(2); k++){
//                 for(NN::Index n = 0; n < output.dimension(2); n++){
                    
//                     output(i, j, k, n) = average;
                    
//                 }
//             }
            
//         }
//     }    
// }


void NN::average_pooling(const Tensor<float, 4> input,
                         Tensor<float, 4> output) {

    assert(input.dimension(0) == output.dimension(0));
    assert(input.dimension(1) == output.dimension(1));

    for(NN::Index n = 0; n < input.dimension(0); n++){
        std::cerr << "n:" << n << std::endl;
        for(NN::Index c = 0; c < input.dimension(1); c++){
            
            float sum = 0.0;
            const NN::Index count = input.dimension(2) * input.dimension(3);
            
            for(NN::Index h = 0; h < input.dimension(2); h++){
                for(NN::Index w = 0; w < input.dimension(3); w++){

                    float val = input(n, c, h, w);
                    sum += val;
                    
                }
                std::cerr << std::endl;
            }
            
            const float average = sum / count;

            
            for(NN::Index h = 0; h < output.dimension(2); h++){
                for(NN::Index w = 0; w < output.dimension(3); w++){
                    
                    output(n, c, h, w) = average;
                    
                }
            }
            
        }
    }    
}

