#ifndef _NNFC_CXXAPI_H
#define _NNFC_CXXAPI_H

#include <any>
#include <vector>

#include "tensor.hh"

namespace nnfc {
namespace cxxapi {

typedef std::vector<std::pair<std::string, std::any>> constructor_list;

typedef std::reference_wrapper<const std::type_info> TypeInfoRef;
typedef std::vector<std::pair<std::string, TypeInfoRef>> constructor_type_list;

// general encoder and decoder interfaces
class EncoderContextInterface {
 public:
  virtual ~EncoderContextInterface() {}
  virtual std::vector<uint8_t> forward(const nn::Tensor<float, 3> input) = 0;
  virtual nn::Tensor<float, 3> backward(
      const nn::Tensor<float, 3> gradient_of_output) = 0;
};

class DecoderContextInterface {
 public:
  virtual ~DecoderContextInterface() {}
  virtual nn::Tensor<float, 3> forward(const std::vector<uint8_t> input) = 0;
  virtual nn::Tensor<float, 3> backward(
      const nn::Tensor<float, 3> gradient_of_output) = 0;
};

// factory functions
std::vector<std::string> get_available_encoders();
std::vector<std::string> get_available_decoders();

constructor_type_list get_encoder_constructor_types(std::string encoder_name);
constructor_type_list get_decoder_constructor_types(std::string decoder_name);

std::unique_ptr<EncoderContextInterface> new_encoder(
    std::string encoder_name, constructor_list constructor_parameters);
std::unique_ptr<DecoderContextInterface> new_decoder(
    std::string decoder_name, constructor_list constructor_parameters);
}
}

#endif  // _NNFC_CXXAPI_H
