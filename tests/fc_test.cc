#include <H5Cpp.h>
#include <iostream>

#include <memory>

#include "tensor.hh"
#include "blob2d.hh"
#include "blob4d.hh"
#include "fullyconnected.hh"

const double tolerance = 1e-6;

int main(int argc, char* argv[]){

    if(argc != 2){
        throw std::runtime_error(std::string("usage: ") + argv[1]  + "<input h5file>");
    }
    
    H5::H5File test_file(argv[1], H5F_ACC_RDONLY);

    H5::DataSet input = test_file.openDataSet("input");
    H5::DataSet weights = test_file.openDataSet("weights");
    H5::DataSet output = test_file.openDataSet("output");

    size_t input_ndims = input.getSpace().getSimpleExtentNdims();
    size_t output_ndims = output.getSpace().getSimpleExtentNdims();

    assert(input_ndims == output_ndims);
    assert(input_ndims == 4);

    hsize_t input_dims[4];
    hsize_t output_dims[4];
    input.getSpace().getSimpleExtentDims(input_dims, NULL);
    output.getSpace().getSimpleExtentDims(output_dims, NULL);

    size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    size_t output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];

    size_t weights_size = input_dims[1] * output_dims[1];
    
    std::unique_ptr<float> input_data(new float[input_size]);
    std::unique_ptr<float> weights_data(new float[weights_size]);

    std::unique_ptr<float> output_data(new float[output_size]);
    std::unique_ptr<float> output_data_correct(new float[output_size]);


    for(size_t i = 0; i < output_size; i++) {
        output_data.get()[i] = i; 
    }

    // load the input and correct result
    input.read(input_data.get(), H5::PredType::NATIVE_FLOAT);
    weights.read(weights_data.get(), H5::PredType::NATIVE_FLOAT);
    output.read(output_data_correct.get(), H5::PredType::NATIVE_FLOAT);

    NN::Tensor<float, 4> input_blob{input_data.get(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    NN::Tensor<float, 2> weights_blob{weights_data.get(), output_dims[1], input_dims[1]};

    NN::Tensor<float, 4> output_blob{output_data.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};

    NN::Tensor<float, 4> output_blob_correct{output_data_correct.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};

    NN::fully_connected(input_blob, weights_blob, output_blob);

    // check output blob
    for(size_t i = 0; i < output_size; i++) {
        const double error = output_data.get()[i] - output_data_correct.get()[i];
        const double squared_error = error*error;
        if( squared_error >= tolerance  ){
            std::cerr << "expected:" << output_data_correct.get()[i] << " but got (C++ code computed):" << output_data.get()[i] << std::endl; 
            throw std::runtime_error("There was a discrepancy between the PyTorch and the nnfc output.");
        }
    }
    
    return 0;
    
}
