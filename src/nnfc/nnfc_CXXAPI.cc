#include <any>
#include <array>
#include <cstdint>
#include <typeinfo>
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "nnfc_CXXAPI.hh"

/*
// This is a helper class for exporting 'layer' implementations for
// the layer factory. 
template<class ContextType, size_t num_constructor_args, typename... constructor_args_types>
class ContextContainer : public nnfc::cxxapi::ContextInterface {
private:

public:
    std::unique_ptr<ContextType> context_;

    template<std::size_t... Idx>
    ContextContainer(std::vector<std::pair<std::string,std::any>> initialization_params,
                            std::index_sequence<Idx...>) :
        context_()
    {
        auto types = ContextContainer::initialization_params();

        for(size_t i = 0; i < num_constructor_args; i++) {
            if(types[i].second.get() != initialization_params[i].second.type()) {
                std::stringstream ss;
                ss << "The type of '" << types[i].first << "' (#" << i << ") was '" << types[i].second.get().name() << "' but excepted '" << initialization_params[i].second.type().name() << "'.";  
                throw std::runtime_error(std::string("Type mismatch during construction of '") + ContextType::name + "'. " + ss.str());
            }
        }
        
        context_ = std::make_unique<ContextType>(std::any_cast< std::tuple_element_t<Idx, std::tuple<constructor_args_types...>>>(initialization_params[Idx].second)...);
    }

    ContextContainer(std::vector<std::pair<std::string, std::any>> initialization_params) :
        ContextContainer(initialization_params, std::make_index_sequence<num_constructor_args>{})
    { }
    
    ~ContextContainer() { }

    // std::vector<uint8_t> forward(nn::Tensor<float, 3> input)
    // {
    //     return context_.encode(input);
    // }

    // nn::Tensor<float, 3> backwards(nn::Tensor<float, 3> gradient_of_output)
    // {
    //     return context_.backwards(gradient_of_output);
    // }

    static std::vector<std::pair<std::string, TypeInfoRef>> initialization_params()
    {
        return ContextType::initialization_params();
    }
};

template<class ContextType, size_t num_constructor_args, typename... constructor_args_types>
struct LayerFactory {

    const std::string name_;

    std::unique_ptr<EncoderContextInterface> new_layer(constructor_list constructor_parameters)
    {
        // static_unique_pointer_cast<EncoderContextInterface>();
        return std::make_unique<ContextContainer<ContextType, num_constructor_args, constructor_args_types...>>(name_, constructor_parameters);
    }

    constructor_type_list get_layer_constructor_types()
    {
        return ContextType::constructor_types();
    }
    
    LayerFactory() :
        name_(ContextType::name),
    { }
};

class Test {
public:
    static const std::string name;

    Test(int x, double y) 
    //Test(std::any x, std::any y) 
    {
        // std::unordered_map<TypeInfoRef, std::string, Hasher, EqualTo> type_names;

        // type_names[typeid(int)] = "int";
        // type_names[typeid(float)] = "float";
        // type_names[typeid(double)] = "double";
        
        // std::cout << type_names[x.type()] << std::endl;
        // std::cout << type_names[y.type()] << std::endl;

        // std::cout << std::any_cast<int>(x) << std::endl;
        // std::cout << std::any_cast<double>(y) << std::endl;
        std::cout << x << " " << y << std::endl;
    }

    
    static std::vector<std::pair<std::string, TypeInfoRef>> initialization_params()
    {
        std::vector<std::pair<std::string, TypeInfoRef>> init_params;

        init_params.push_back(std::pair<std::string, TypeInfoRef>("magic_num", typeid(int)));
        init_params.push_back(std::pair<std::string, TypeInfoRef>("magic_float", typeid(double)));

        return init_params;
    }    
};
const std::string Test::name = "test";

*/
