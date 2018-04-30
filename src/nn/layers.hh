#ifndef _NN_LAYERS_H
#define _NN_LAYERS_H

#include <H5Cpp.h>
#include <memory>

#include "tensor.hh"
#include "activation.hh"
#include "convolution.hh"
#include "fullyconnected.hh"
#include "normalization.hh"
#include "pool.hh"

namespace nn {

    class LayerInterface
    {
    public:
        virtual ~LayerInterface() { };
        virtual Tensor<float, 4> forward(const Tensor<float, 4> input) = 0;
    };

    class ConvolutionLayer : public LayerInterface
    {
    private:
        Tensor<float, 4> output_;
        const Tensor<float, 4> kernel_;
        const size_t stride_;
        const size_t zero_padding_;
        
    public:
        ConvolutionLayer(Tensor<float, 4> output,
                         const Tensor<float, 4> kernel,
                         const size_t stride,
                         const size_t zero_padding) :
            output_(output),
            kernel_(kernel),
            stride_(stride),
            zero_padding_(zero_padding)
        { }

        ~ConvolutionLayer() { }

        Tensor<float, 4> forward(const Tensor<float, 4> input)
        {
            conv2d(input, kernel_, output_, stride_, zero_padding_);
            return output_;
        }
    };
        
    class FCLayer : public LayerInterface
    {
    private:
        Tensor<float, 4> output_;
        const Tensor<float, 2> weights_;
        
    public:
        FCLayer(Tensor<float, 4> output, Tensor<float, 2> weights) :
            output_(output),
            weights_(weights)
        { }

        ~FCLayer() { }

        Tensor<float, 4> forward(const Tensor<float, 4> input)
        {
            fully_connected(input, weights_, output_);
            return output_;
        }
    };

    class FCWithBiasLayer : public LayerInterface
    {
    private:
        Tensor<float, 4> output_;
        const Tensor<float, 2> weights_;
        const Tensor<float, 1> bias_;
        
    public:
        FCWithBiasLayer(Tensor<float, 4> output, Tensor<float, 2> weights, Tensor<float, 1> bias) :
            output_(output),
            weights_(weights),
            bias_(bias)
        { }

        ~FCWithBiasLayer() { }

        Tensor<float, 4> forward(const Tensor<float, 4> input)
        {
            fully_connected_with_bias(input, weights_, bias_, output_);
            return output_;
        }
    };
    
    class BatchNormLayer : public LayerInterface
    {
    private:
        Tensor<float, 4> output_;
        const Tensor<float, 1> means_;
        const Tensor<float, 1> variances_;
        const Tensor<float, 1> weight_;
        const Tensor<float, 1> bias_;
        const float eps_;
        
    public:
        BatchNormLayer(Tensor<float, 4> output,
                       const Tensor<float, 1> means,
                       const Tensor<float, 1> variances,
                       const Tensor<float, 1> weight,
                       const Tensor<float, 1> bias,
                       const float eps) :
            output_(output),
            means_(means),
            variances_(variances),
            weight_(weight),
            bias_(bias),
            eps_(eps)
        { }

        ~BatchNormLayer() { }

        Tensor<float, 4> forward(const Tensor<float, 4> input)
        {
            batch_norm(input, means_, variances_, weight_, bias_, output_, eps_);
            return output_;
        }
    };
    
    class ReluLayer : public LayerInterface
    {
    private:
        Tensor<float, 4> output_;
        
    public:
        ReluLayer(Tensor<float, 4> output) :
            output_(output)
        { }

        ~ReluLayer() { }

        Tensor<float, 4> forward(const Tensor<float, 4> input)
        {
            relu(input, output_);
            return output_;
        }
    };

    class PoolLayer : public LayerInterface
    {
    private:
        Tensor<float, 4> output_;
        
    public:
        PoolLayer(Tensor<float, 4> output) :
            output_(output)
        { }

        ~PoolLayer() { }

        Tensor<float, 4> forward(const Tensor<float, 4> input)
        {
            average_pooling(input, output_);
            return output_;
        }
    };
    
    std::shared_ptr<LayerInterface> make_convolution_from_hdf5(size_t output_batch_size,
                                                               size_t output_channels,
                                                               size_t output_height,
                                                               size_t output_width,
                                                               H5::H5File weights_file,
                                                               std::string kernel_name,
                                                               size_t stride,
                                                               size_t zero_padding);
    
    std::shared_ptr<LayerInterface> make_fc_from_hdf5(size_t output_batch_size,
                                                      size_t output_channels,
                                                      size_t output_height,
                                                      size_t output_width,
                                                      H5::H5File weights_file,
                                                      std::string kernel_name);
            
    std::shared_ptr<LayerInterface> make_fc_with_bias_from_hdf5(size_t output_batch_size,
                                                                size_t output_channels,
                                                                size_t output_height,
                                                                size_t output_width,
                                                                H5::H5File weights_file,
                                                                std::string kernel_name,
                                                                std::string bias_name);

    std::shared_ptr<LayerInterface> make_batch_norm_from_hdf5(size_t output_batch_size,
                                                              size_t output_channels,
                                                              size_t output_height,
                                                              size_t output_width,
                                                              H5::H5File weights_file,
                                                              std::string means_name,
                                                              std::string variances_name,
                                                              std::string weight_name,
                                                              std::string bias_name,
                                                              float eps);
    
    std::shared_ptr<LayerInterface> make_relu_from_hdf5(size_t output_batch_size,
                                                        size_t output_channels,
                                                        size_t output_height,
                                                        size_t output_width);
    

    std::shared_ptr<LayerInterface> make_pool_from_hdf5(size_t output_batch_size,
                                                        size_t output_channels,
                                                        size_t output_height,
                                                        size_t output_width);
    
}

#endif // _NN_LAYERS_H
