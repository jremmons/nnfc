#include <H5Cpp.h>

#include <iostream>

int main(int argc, char* argv[]){

    H5::H5File file("/tmp/test.h5", H5F_ACC_TRUNC);

    
    
    std::cout << argc << " " << argv[1] << "\n";
    return 0;
    
}
