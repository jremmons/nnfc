#include <H5Cpp.h>
#include <iostream>

#include <memory>

#include "tensor.hh"
#include "normalization.hh"

const double tolerance = 1e-6;

int main(int argc, char* argv[]){

    if(argc != 2){
        throw std::runtime_error(std::string("usage: ") + argv[1]  + "<input h5file>");
    }
    
    H5::H5File test_file(argv[1], H5F_ACC_RDONLY);

    H5::DataSet input = test_file.openDataSet("input");
    H5::DataSet means = test_file.openDataSet("means");
    H5::DataSet variances = test_file.openDataSet("variances");
    H5::DataSet weight = test_file.openDataSet("weight");
    H5::DataSet bias = test_file.openDataSet("bias");
    H5::DataSet epsilon = test_file.openDataSet("eps");
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

    size_t bn_size = output_dims[1];
    
    std::unique_ptr<float[]> input_data(new float[input_size]);
    std::unique_ptr<float[]> means_data(new float[bn_size]);
    std::unique_ptr<float[]> variances_data(new float[bn_size]);

    std::unique_ptr<float[]> weight_data(new float[bn_size]);
    std::unique_ptr<float[]> bias_data(new float[bn_size]);

    std::unique_ptr<float[]> output_data(new float[output_size]);
    std::unique_ptr<float[]> output_data_correct(new float[output_size]);

    for(size_t i = 0; i < input_size; i++) {
        output_data[i] = i; 
    }

    // load the input and correct result
    input.read(input_data.get(), H5::PredType::NATIVE_FLOAT);
    means.read(means_data.get(), H5::PredType::NATIVE_FLOAT);
    variances.read(variances_data.get(), H5::PredType::NATIVE_FLOAT);
    weight.read(weight_data.get(), H5::PredType::NATIVE_FLOAT);
    bias.read(bias_data.get(), H5::PredType::NATIVE_FLOAT);

    float epsilon_val;
    epsilon.read(&epsilon_val, H5::PredType::NATIVE_FLOAT);
    
    output.read(output_data_correct.get(), H5::PredType::NATIVE_FLOAT);

    nn::Tensor<float, 4> input_blob{input_data.get(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]};

    nn::Tensor<float, 1> means_blob{means_data.get(), input_dims[1]};
    nn::Tensor<float, 1> variances_blob{variances_data.get(), input_dims[1]};

    nn::Tensor<float, 1> weight_blob{weight_data.get(), input_dims[1]};
    nn::Tensor<float, 1> bias_blob{bias_data.get(), input_dims[1]};

    nn::Tensor<float, 4> output_blob{output_data.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
    nn::Tensor<float, 4> output_blob_correct{output_data_correct.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};
    
    nn::batch_norm(input_blob, means_blob, variances_blob, weight_blob, bias_blob, output_blob, epsilon_val);

    // check output blob
    for(size_t i = 0; i < input_size; i++) {
        const float error = output_data[i] - output_data_correct[i];
        const float squared_error = error*error;
        if( squared_error > tolerance ){
            std::cerr << "expected:" << output_data_correct.get()[i] << " but got computed:" << output_data[i] << "\n"; 
            throw std::runtime_error("There was a discrepancy between the PyTorch and the nnfc output.");
        }
    }
    
    return 0;
    
}
