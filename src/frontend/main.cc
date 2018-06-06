#include <any>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>

using TypeInfoRef = std::reference_wrapper<const std::type_info>;

struct Hasher {
  std::size_t operator()(TypeInfoRef code) const {
    return code.get().hash_code();
  }
};

struct EqualTo {
  bool operator()(TypeInfoRef lhs, TypeInfoRef rhs) const {
    return lhs.get() == rhs.get();
  }
};

#include "nnfc.hh"
#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

template <class EncoderContextType, size_t num_constructor_args,
          typename... constructor_args_types>
class EncoderContextContainer : public nnfc::cxxapi::EncoderContextInterface {
 private:
 public:
  std::unique_ptr<EncoderContextType> context_;

  template <std::size_t... Idx>
  EncoderContextContainer(
      std::vector<std::pair<std::string, std::any>> initialization_params,
      std::index_sequence<Idx...>)
      : context_() {
    auto types = EncoderContextContainer::initialization_params();

    for (size_t i = 0; i < num_constructor_args; i++) {
      if (types[i].second.get() != initialization_params[i].second.type()) {
        std::stringstream ss;
        ss << "The type of '" << types[i].first << "' (#" << i << ") was '"
           << types[i].second.get().name() << "' but excepted '"
           << initialization_params[i].second.type().name() << "'.";
        throw std::runtime_error(
            std::string("Type mismatch during construction of '") +
            EncoderContextType::name + "'. " + ss.str());
      }
    }

    context_ = std::make_unique<EncoderContextType>(
        std::any_cast<
            std::tuple_element_t<Idx, std::tuple<constructor_args_types...>>>(
            initialization_params[Idx].second)...);
  }

  EncoderContextContainer(
      std::vector<std::pair<std::string, std::any>> initialization_params)
      : EncoderContextContainer(
            initialization_params,
            std::make_index_sequence<num_constructor_args>{}) {}

  ~EncoderContextContainer() {}

  // std::vector<uint8_t> encode(nn::Tensor<float, 3> input)
  // {
  //     return context_.encode(input);
  // }

  // nn::Tensor<float, 3> backwards(nn::Tensor<float, 3> gradient_of_output)
  // {
  //     return context_.backwards(gradient_of_output);
  // }

  static std::vector<std::pair<std::string, TypeInfoRef>>
  initialization_params() {
    return EncoderContextType::initialization_params();
  }
};

class Test {
 public:
  static const std::string name;

  Test(int x, double y)
  // Test(std::any x, std::any y)
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

  static std::vector<std::pair<std::string, TypeInfoRef>>
  initialization_params() {
    std::vector<std::pair<std::string, TypeInfoRef>> init_params;

    init_params.push_back(
        std::pair<std::string, TypeInfoRef>("magic_num", typeid(int)));
    init_params.push_back(
        std::pair<std::string, TypeInfoRef>("magic_float", typeid(double)));

    return init_params;
  }
};
const std::string Test::name = "test";

int main(int, char**) {
  // std::tuple<std::any, std::any> init_list = { 42, 3.14 };
  std::vector<std::pair<std::string, std::any>> init_list = {
      std::make_pair<std::string, std::any>("intval", 42),
      std::make_pair<std::string, std::any>("doubleval", 3.14f)};

  // std::cout << std::any_cast<>(std::get<0>(init_list)) << "\n";
  // std::cout << std::any_cast<int>(std::get<1>(init_list)) << "\n";

  EncoderContextContainer<Test, 2, int, double> container(init_list);

  // std::cout << container.context_.get()->x_ << "\n";
  // std::cout << container.context_.get()->y_ << "\n";

  // std::cout << argc << " " << argv[0] << "\n";

  // float *data = new float[sizeof(float) * 100 * 32 * 32 * 3];

  // nn::Tensor<float, 4> tensor(data, 100, 32, 32, 3);

  // std::cout << tensor.dimension(0) << std::endl;
  // std::cout << tensor.dimension(1) << std::endl;
  // std::cout << tensor.dimension(2) << std::endl;
  // std::cout << tensor.dimension(3) << std::endl;

  // std::cout << tensor.size() << std::endl;
  // std::cout << tensor.rank() << std::endl;

  // std::cout << tensor(0,0,1,3) << std::endl;

  // nn::Tensor<float, 2> new_tensor(data, 32, 32);
  // // std::cout << new_tensor.dimension(0) << std::endl;
  // // std::cout << new_tensor.dimension(1) << std::endl;

  // nn::Tensor<float, 2> create_tensor(32, 32);
  // std::cout << create_tensor.dimension(0) << std::endl;
  // std::cout << create_tensor.dimension(1) << std::endl;

  // auto move_tensor = std::move(create_tensor);
  // std::cout << move_tensor.dimension(0) << std::endl;
  // std::cout << move_tensor.dimension(1) << std::endl;

  // auto eq_tensor = move_tensor;
  // std::cout << eq_tensor.dimension(0) << std::endl;
  // std::cout << eq_tensor.dimension(1) << std::endl;

  // tensor(0,0,0,0) = -1;
  // auto shallowcopy = tensor;
  // auto deepcopy = tensor.deepcopy();

  // shallowcopy(0,0,0,0) = 17;
  // deepcopy(0,0,0,0) = 42;

  // std::cout << tensor(0,0,0,0) << std::endl;
  // std::cout << shallowcopy(0,0,0,0) << std::endl;
  // std::cout << deepcopy(0,0,0,0) << std::endl;

  // auto copy_tensor = create_tensor;
  // forward pass of nn
  // todo add code here
  // Blob3D<float> kernel_weights{nullptr, 0, 0, 0};
  // Blob4D<float> inputs{nullptr, 0, 0, 0, 0};
  // Blob4D<float> outputs{nullptr, 0, 0, 0, 0};

  // nn::forward(images, predictions);

  return 0;
}
