#include <H5Cpp.h>

#include <iostream>
#include <stdexcept>

#include "convolution.hh"

const double tolerance = 1e-6;

int main(int argc, char* argv[]){

    if(argc != 2){
        throw std::runtime_error(std::string("usage: ") + argv[1]  + "<input h5file>");
    }
    
    H5::H5File test_file(argv[1], H5F_ACC_RDONLY);

    H5::DataSet input = test_file.openDataSet("input");
    H5::DataSet kernel = test_file.openDataSet("kernel");
    H5::DataSet stride = test_file.openDataSet("stride");
    H5::DataSet zero_padding = test_file.openDataSet("zero_padding");
    H5::DataSet output = test_file.openDataSet("output");

    size_t input_ndims = input.getSpace().getSimpleExtentNdims();
    size_t kernel_ndims = kernel.getSpace().getSimpleExtentNdims();
    size_t output_ndims = output.getSpace().getSimpleExtentNdims();
    
    assert(input_ndims == output_ndims);
    assert(input_ndims == 4);
    assert(kernel_ndims == 4);
    
    hsize_t input_dims[4];
    hsize_t output_dims[4];
    input.getSpace().getSimpleExtentDims(input_dims, NULL);
    output.getSpace().getSimpleExtentDims(output_dims, NULL);

    size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    size_t output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    
    hsize_t kernel_dims[4];
    kernel.getSpace().getSimpleExtentDims(kernel_dims, NULL);

    size_t kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2] * kernel_dims[3];

    std::unique_ptr<float> input_data(new float[input_size]);
    std::unique_ptr<float> kernel_data(new float[kernel_size]);
    
    std::unique_ptr<float> output_data(new float[output_size]);
    std::unique_ptr<float> output_data_correct(new float[output_size]);

    for(size_t i = 0; i < output_size; i++) {
        output_data.get()[i] = i; 
    }

    // load the input and correct result
    input.read(input_data.get(), H5::PredType::NATIVE_FLOAT);
    kernel.read(kernel_data.get(), H5::PredType::NATIVE_FLOAT);

    uint64_t stride_;
    stride.read(&stride_, H5::PredType::NATIVE_UINT64);
    
    uint64_t zero_padding_;
    zero_padding.read(&zero_padding_, H5::PredType::NATIVE_UINT64);

    output.read(output_data_correct.get(), H5::PredType::NATIVE_FLOAT);

    Blob4D<float> input_blob{input_data.get(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    Blob4D<float> kernel_blob{kernel_data.get(), kernel_dims[0], kernel_dims[1], kernel_dims[2], kernel_dims[3]};

    Blob4D<float> output_blob{output_data.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
    Blob4D<float> output_blob_correct{output_data_correct.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
    
    NN::conv2d(input_blob, kernel_blob, output_blob, stride_, zero_padding_);

    // check output blob
    for(size_t i = 0; i < output_size; i++) {
        std::cerr << output_data_correct.get()[i] << std::endl;
        std::cerr << output_data.get()[i] << std::endl;

        const float error = output_data.get()[i] - output_data_correct.get()[i];
        const float squared_error = error*error;
        if( squared_error > tolerance ){
            std::cerr << "expected:" << output_data_correct.get()[i] << " but got computed:" << output_data.get()[i] << "\n"; 
            throw std::runtime_error("There was a discrepancy between the PyTorch and the nnfc output.");
        }
    }
    
    return 0;
    
}
