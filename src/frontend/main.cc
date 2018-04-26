#include <iostream>

// #include "blob1d.hh"
// #include "blob3d.hh"
// #include "blob4d.hh"

#include "tensor.hh"

#include "nn.hh"
#include "nnfc.hh"

int main(int argc, char* argv[]){

    std::cout << argc << " " << argv[0] << "\n";

    float *data = new float[sizeof(float) * 100 * 32 * 32 * 3];

    NNFC::Tensor<float, 4> tensor(data, 100, 32, 32, 3);

    std::cout << tensor.dimension(0) << std::endl;
    std::cout << tensor.dimension(1) << std::endl;
    std::cout << tensor.dimension(2) << std::endl;
    std::cout << tensor.dimension(3) << std::endl;

    std::cout << tensor.size() << std::endl;
    std::cout << tensor.rank() << std::endl;    

    std::cout << tensor(0,0,1,3) << std::endl;
    
    NNFC::Tensor<float, 2> new_tensor(data, 32, 32);
    // std::cout << new_tensor.dimension(0) << std::endl;
    // std::cout << new_tensor.dimension(1) << std::endl;

    NNFC::Tensor<float, 2> create_tensor(32, 32);
    std::cout << create_tensor.dimension(0) << std::endl;
    std::cout << create_tensor.dimension(1) << std::endl;

    auto move_tensor = std::move(create_tensor);
    std::cout << move_tensor.dimension(0) << std::endl;
    std::cout << move_tensor.dimension(1) << std::endl;

    //auto copy_tensor = create_tensor;
    // forward pass of nn
    // todo add code here
    //Blob3D<float> kernel_weights{nullptr, 0, 0, 0};
    //Blob4D<float> inputs{nullptr, 0, 0, 0, 0};
    //Blob4D<float> outputs{nullptr, 0, 0, 0, 0};

    //NN::forward(images, predictions);
    
    return 0;
}
