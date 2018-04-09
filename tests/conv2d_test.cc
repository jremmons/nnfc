#include <H5Cpp.h>

#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]){

    if(argc != 2){
        throw std::runtime_error(std::string("usage: ") + argv[1]  + "<input h5file>");
    }
    
    H5::H5File test_file(argv[1], H5F_ACC_RDONLY);

    H5::DataSet input = test_file.openDataSet("input");
    H5::DataSet kernel = test_file.openDataSet("kernel");
    H5::DataSet stride = test_file.openDataSet("stride");
    H5::DataSet padding = test_file.openDataSet("padding");
    H5::DataSet output = test_file.openDataSet("output");
    
    return 0;
    
}
