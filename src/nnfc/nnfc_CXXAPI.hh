#ifndef _NNFC_CXXAPI_H
#define _NNFC_CXXAPI_H

#include <vector>

#include "nnfc.hh"
#include "tensor.hh"

namespace nnfc { namespace cxxapi {
        
        // TODO(jremmons) eventually we will want to create a standardized way
        // for external code projects (including our own python API) to
        // interface with our code. This header file is my first attempt at
        // outlining a possible API for external projects.         

        // typedef std::vector<std::pair<std::string, std::any>> constructor_list;
        // typedef std::vector<std::pair<std::string, TypeInfoRef>> constructor_type_list;
        
        class EncoderContextInterface {
        public:
            virtual ~EncoderContextInterface() { }
            // virtual std::vector<uint8_t> forward(nn::Tensor<float, 3> input) = 0;
            // virtual nn::Tensor<float, 3> backward(nn::Tensor<float, 3> gradient_of_output) = 0;

            // virtual std::string name() = 0;
            // virtual std::string docs() = 0;
            
            // static std::vector<std::pair<std::string, std::type_info>> ();
        };

        // std::vector<std::string> get_available_layers();

        // constructor_type_list get_layer_constructor_types_by_name(std::string codec_name);

        // std::unique_ptr<EncoderContextInterface> get_layer_by_name(std::string codec_name, constructor_list constructor_parameters);
    }
}
#endif // _NNFC_CXXAPI_H
