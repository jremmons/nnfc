#include <H5Cpp.h>
#include <iostream>

#include <memory>

#include "tensor.hh"
#include "activation.hh"

int main(int argc, char* argv[]){

    if(argc != 2){
        throw std::runtime_error(std::string("usage: ") + argv[1]  + "<input h5file>");
    }
    
    H5::H5File test_file(argv[1], H5F_ACC_RDONLY);

    H5::DataSet input = test_file.openDataSet("input");
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

    std::unique_ptr<float[]> input_data(new float[input_size]);
    std::unique_ptr<float[]> output_data(new float[output_size]);
    std::unique_ptr<float[]> output_data_correct(new float[output_size]);

    for(size_t i = 0; i < input_size; i++) {
        output_data.get()[i] = i; 
    }

    // load the input and correct result
    input.read(input_data.get(), H5::PredType::NATIVE_FLOAT);
    output.read(output_data_correct.get(), H5::PredType::NATIVE_FLOAT);

    nn::Tensor<float, 4> input_blob{input_data.get(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    nn::Tensor<float, 4> output_blob{output_data.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};

    nn::Tensor<float, 4> output_blob_correct{output_data_correct.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
    
    nn::relu(input_blob, output_blob); // run the actication function

    // check output blob
    for(size_t i = 0; i < input_size; i++) {
        if(output_data[i] != output_data_correct[i]){
            throw std::runtime_error("There was a discrepancy between the PyTorch and the nnfc output.");
        }
    }
    
    return 0;
    
}
