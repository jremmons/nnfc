#include <vector>
#include <memory>

#include "net.hh"
#include "tensor.hh"
#include "layers.hh"

nn::Net::Net() :
    layers_()
{ }

nn::Net::Net(std::vector<std::shared_ptr<nn::LayerInterface>> layers) :
    layers_(layers)
{ }

nn::Net::~Net() { }

nn::Net nn::Net::operator+=(std::shared_ptr<nn::LayerInterface> layer)
{
    layers_.push_back(layer);
    return *this;
}

nn::Tensor<float, 4> nn::Net::forward(nn::Tensor<float, 4> input)
{

    std::vector<nn::Tensor<float, 4>> outputs;
    outputs.push_back(input);
    
    for(auto layer : layers_)
    {
        auto input = outputs[outputs.size() - 1];

        auto output = layer.get()->forward(input);

        outputs.push_back(output);
    }
    
    return outputs[outputs.size() - 1];
}
