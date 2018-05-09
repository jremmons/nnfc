#include <any>
#include <array>
#include <cstdint>
#include <typeinfo>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include "nnfc.hh"
#include "nnfc_CXXAPI.hh"

//////////////////////////////////////////////////////////////////////
//
// helper classes
//
//////////////////////////////////////////////////////////////////////
template<class ContextInterface, class ContextType, class input_T, class output_T, typename... constructor_args_types>
class ContextContainer : public ContextInterface {
// ContextContainer: a helper class for exporting contexts (or NN
// layers) so they all have a common interface. This class shouldn't
// be used outside of this file. To see the API, look in the header
// file for the definition of a {Encoder, Decoder}ContextInterface.

private:
    static constexpr size_t num_constructor_args = std::tuple_size<std::tuple<constructor_args_types...>>{};

    std::unique_ptr<ContextType> context_;
    
public:

    template<std::size_t... Idx>
    ContextContainer(nnfc::cxxapi::constructor_list initialization_params,
                     std::index_sequence<Idx...>) :
        context_()
    {
        auto types = ContextType::initialization_params();
        
        for(size_t i = 0; i < num_constructor_args; i++) {
            if(types[i].second.get() != initialization_params[i].second.type()) {
                std::stringstream ss;
                ss << "The type of '" <<
                    types[i].first <<
                    "' (#" << i << ") was '" <<
                    types[i].second.get().name() <<
                    "' but excepted '" <<
                    initialization_params[i].second.type().name() << "'."; 

                throw std::runtime_error(std::string("Type mismatch during construction. ") + ss.str());
            }
        }
        
        context_ = std::make_unique<ContextType>(std::any_cast< std::tuple_element_t<Idx, std::tuple<constructor_args_types...>>>(initialization_params[Idx].second)...);
    }

    ContextContainer(nnfc::cxxapi::constructor_list initialization_params) :
        ContextContainer(initialization_params, std::make_index_sequence<num_constructor_args>{})
    { }
    
    ~ContextContainer() { }

    output_T forward(input_T input) override
    {
        return context_->forward(input);
    }

    nn::Tensor<float, 3> backward(nn::Tensor<float, 3> gradient_of_output) override
    {
        return context_->backward(gradient_of_output);
    }
};


template<class EncoderContextType, typename... constructor_args_types>
class EncoderContextContainer : public ContextContainer<nnfc::cxxapi::EncoderContextInterface,
                                                        EncoderContextType,
                                                        nn::Tensor<float, 3>,
                                                        std::vector<uint8_t>,
                                                        constructor_args_types...>
{
public:

    EncoderContextContainer(nnfc::cxxapi::constructor_list initialization_params) :
        ContextContainer<nnfc::cxxapi::EncoderContextInterface,
                         EncoderContextType,
                         nn::Tensor<float, 3>,
                         std::vector<uint8_t>,
                         constructor_args_types...>(initialization_params)
    { }

};


template<class DecoderContextType, typename... constructor_args_types>
class DecoderContextContainer : public ContextContainer<nnfc::cxxapi::DecoderContextInterface,
                                                        DecoderContextType,
                                                        std::vector<uint8_t>,
                                                        nn::Tensor<float, 3>,
                                                        constructor_args_types...> {
public:

    DecoderContextContainer(nnfc::cxxapi::constructor_list initialization_params) :
        ContextContainer<nnfc::cxxapi::DecoderContextInterface,
                         DecoderContextType,
                         std::vector<uint8_t>,
                         nn::Tensor<float, 3>,
                         constructor_args_types...>(initialization_params)
    { }

};


//////////////////////////////////////////////////////////////////////
//
// factory functions
//
//////////////////////////////////////////////////////////////////////
template<class ContextType, typename... constructor_args_types>
static std::unique_ptr<nnfc::cxxapi::EncoderContextInterface> new_encoder(nnfc::cxxapi::constructor_list initialization_params)
{
    return std::make_unique<EncoderContextContainer<ContextType, constructor_args_types...>>(initialization_params);
}

template<class ContextType, typename... constructor_args_types>
static std::unique_ptr<nnfc::cxxapi::DecoderContextInterface> new_decoder(nnfc::cxxapi::constructor_list initialization_params)
{
    return std::make_unique<DecoderContextContainer<ContextType, constructor_args_types...>>(initialization_params);
}

template<class ContextType>
static nnfc::cxxapi::constructor_type_list constructor_types()
{
    return ContextType::initialization_params();
}

struct EncoderContextFactory
{
    const std::string exported_name;
    std::function<std::unique_ptr<nnfc::cxxapi::EncoderContextInterface>(nnfc::cxxapi::constructor_list)> new_context_func;
    std::function<nnfc::cxxapi::constructor_type_list()> constructor_types_func;
};

struct DecoderContextFactory
{
    const std::string exported_name;
    std::function<std::unique_ptr<nnfc::cxxapi::DecoderContextInterface>(nnfc::cxxapi::constructor_list)> new_context_func;
    std::function<nnfc::cxxapi::constructor_type_list()> constructor_types_func;
};


//////////////////////////////////////////////////////////////////////
//
// The std::vectors below define the available exported codecs.
// Add any new codecs to these arrays to export them. 
//
//////////////////////////////////////////////////////////////////////
static const std::string test_name = "__test";

static std::vector<EncoderContextFactory> nnfc_available_encoders = {
    {
        .exported_name = "simple_encoder",
        .new_context_func = new_encoder<nnfc::SimpleEncoder>,
        .constructor_types_func = constructor_types<nnfc::SimpleEncoder>
    }
};

static std::vector<DecoderContextFactory> nnfc_available_decoders = {
    // {
    //     .exported_name = test_name.c_str(),
    //     .new_context_func = new_decoder<TestDec, int, double>,
    //     .constructor_types_func = no_constructor_args
    // }
};


//////////////////////////////////////////////////////////////////////
//
// Helper functions.
//
//////////////////////////////////////////////////////////////////////
static EncoderContextFactory get_encoder_factory(std::string encoder_name)
{
    for(size_t i = 0; i < nnfc_available_encoders.size(); i++) {
        if(nnfc_available_encoders[i].exported_name == encoder_name) {
            return nnfc_available_encoders[i];
        }
    }

    throw std::runtime_error(encoder_name + " is not available.");
}

static DecoderContextFactory get_decoder_factory(std::string decoder_name)
{
    for(size_t i = 0; i < nnfc_available_decoders.size(); i++) {
        if(nnfc_available_decoders[i].exported_name == decoder_name) {
            return nnfc_available_decoders[i];
        }
    }

    throw std::runtime_error(decoder_name + " is not available.");
}


//////////////////////////////////////////////////////////////////////
//
// Public functions exported by this file. (see below)
//
//////////////////////////////////////////////////////////////////////
std::vector<std::string> nnfc::cxxapi::get_available_encoders()
{
    std::vector<std::string> available_encoders;
    for(size_t i = 0; i < nnfc_available_encoders.size(); i++) {
        std::string context_name = nnfc_available_encoders[i].exported_name;
        
        if(context_name != test_name) {
            available_encoders.push_back(nnfc_available_encoders[i].exported_name);
        }
    }

    return available_encoders;
}

std::vector<std::string> nnfc::cxxapi::get_available_decoders()
{
    std::vector<std::string> available_decoders;
    for(size_t i = 0; i < nnfc_available_decoders.size(); i++) {
        std::string context_name = nnfc_available_decoders[i].exported_name;
        
        if(context_name != test_name) {
            available_decoders.push_back(nnfc_available_decoders[i].exported_name);
        }
    }

    return available_decoders;
}

nnfc::cxxapi::constructor_type_list nnfc::cxxapi::get_encoder_constructor_types(std::string encoder_name)
{
    const EncoderContextFactory factory = get_encoder_factory(encoder_name);
    return factory.constructor_types_func();
}

std::unique_ptr<nnfc::cxxapi::EncoderContextInterface> nnfc::cxxapi::new_encoder(std::string encoder_name, nnfc::cxxapi::constructor_list params)
{
    const EncoderContextFactory factory = get_encoder_factory(encoder_name);
    return factory.new_context_func(params);
}

nnfc::cxxapi::constructor_type_list nnfc::cxxapi::get_decoder_constructor_types(std::string decoder_name)
{
    const DecoderContextFactory factory = get_decoder_factory(decoder_name);
    return factory.constructor_types_func();
}

std::unique_ptr<nnfc::cxxapi::DecoderContextInterface> nnfc::cxxapi::new_decoder(std::string decoder_name, nnfc::cxxapi::constructor_list params)
{
    const DecoderContextFactory factory = get_decoder_factory(decoder_name);
    return factory.new_context_func(params);
}
