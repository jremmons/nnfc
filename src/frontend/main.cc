#include <iostream>

#include "blob1d.hh"
#include "blob3d.hh"
#include "blob4d.hh"

#include "nn.hh"
#include "nnfc.hh"

int main(int argc, char* argv[]){

    std::cout << argc << " " << argv[0] << "\n";

    // forward pass of nn
    // todo add code here
    Blob3D<float> kernel_weights{nullptr, 0, 0, 0};
    Blob4D<float> inputs{nullptr, 0, 0, 0, 0};
    Blob4D<float> outputs{nullptr, 0, 0, 0, 0};

    //NN::forward(images, predictions);
    
    return 0;
}
