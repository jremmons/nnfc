#ifndef _NN_NET_H
#define _NN_NET_H

#include <memory>
#include <vector>

#include "layers.hh"
#include "tensor.hh"

namespace nn {

class Net {
 private:
  std::vector<std::shared_ptr<LayerInterface>> layers_;

 public:
  Net();
  Net(std::vector<std::shared_ptr<LayerInterface>> layers);
  ~Net();

  Net operator+=(std::shared_ptr<LayerInterface> layer);
  Tensor<float, 4> forward(Tensor<float, 4> input);
};
}

#endif  // _NN_NET_H
