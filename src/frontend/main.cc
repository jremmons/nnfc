#include <iostream>

#include "nn.hh"
#include "nnfc.hh"

int main(int argc, char* argv[]){

    std::cout << argc << " " << argv[0] << "\n";

    // forward pass of nn
    // todo add code here
    Blob1D<uint64_t> predictions{nullptr, 0};
    Blob4D<float> images{nullptr, 0, 0, 0, 0};

    NN::forward(images, predictions);
    
    return 0;
}
