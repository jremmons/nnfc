#include <H5Cpp.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "tensor.hh"
#include "layers.hh"
#include "net.hh"

const double tolerance = 1e-6;

int main(int argc, char* argv[]){

    if(argc != 2){
        std::cout << "usage: " << argv[0] << " <input h5file>\n";
        return -1;
    }
    
    H5::H5File test_file(argv[1], H5F_ACC_RDONLY);

    // read the input data
    H5::DataSet input = test_file.openDataSet("input");
    H5::DataSet output = test_file.openDataSet("output");

    size_t input_ndims = input.getSpace().getSimpleExtentNdims();
    size_t output_ndims = output.getSpace().getSimpleExtentNdims();
    
    hsize_t input_dims[4];
    hsize_t output_dims[2];
    input.getSpace().getSimpleExtentDims(input_dims, NULL);
    output.getSpace().getSimpleExtentDims(output_dims, NULL);

    size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    size_t output_size = output_dims[0] * output_dims[1];
    
    assert(input_ndims == 4);
    assert(output_ndims == 2);

    std::unique_ptr<float[]> input_data(new float[input_size]);
    std::unique_ptr<float[]> output_data_correct(new float[output_size]);

    input.read(input_data.get(), H5::PredType::NATIVE_FLOAT);
    output.read(output_data_correct.get(), H5::PredType::NATIVE_FLOAT);

    nn::Tensor<float, 4> input_blob(input_data.get(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]);

    // define the network
    nn::Net simple_cnn{};
    simple_cnn += nn::make_convolution_from_hdf5(input_dims[0],
                                                 64,
                                                 input_dims[2],
                                                 input_dims[3],
                                                 test_file,
                                                 "conv1.weight", 1,1);

    simple_cnn += nn::make_batch_norm_from_hdf5(input_dims[0],
                                                64,
                                                input_dims[2],
                                                input_dims[3],
                                                test_file,
                                                "bn1.running_mean",
                                                "bn1.running_var",
                                                "bn1.weight",
                                                "bn1.bias",
                                                0.00001);
    
    simple_cnn += nn::make_relu_from_hdf5(input_dims[0],
                                          64,
                                          input_dims[2],
                                          input_dims[3]);    
    
    simple_cnn += nn::make_pool_from_hdf5(input_dims[0], 64, 1, 1);

    simple_cnn += nn::make_fc_from_hdf5(input_dims[0], 10, 1, 1,
                                        test_file,
                                        "linear.weight");

    // run the network
    auto output_blob = simple_cnn.forward(input_blob);
    
    // std::cout << "output.dim(0) " << output_blob.dimension(0) << std::endl;  
    // std::cout << "output.dim(1) " << output_blob.dimension(1) << std::endl;  
    // std::cout << "output.dim(2) " << output_blob.dimension(2) << std::endl;  
    // std::cout << "output.dim(3) " << output_blob.dimension(3) << std::endl;  

    nn::Tensor<float, 4> output_blob_correct(output_data_correct.get(), output_dims[0], output_dims[1], 1, 1);

    // std::cout << "output_correct.dim(0) " << output_blob_correct.dimension(0) << std::endl;  
    // std::cout << "output_correct.dim(1) " << output_blob_correct.dimension(1) << std::endl;  
    // std::cout << "output_correct.dim(2) " << output_blob_correct.dimension(2) << std::endl;  
    // std::cout << "output_correct.dim(3) " << output_blob_correct.dimension(3) << std::endl;  

    // check output blob
    for(nn::Index n = 0; n < output_blob_correct.dimension(0); n++) {
        for(nn::Index c = 0; c < output_blob_correct.dimension(1); c++) {
            for(nn::Index h = 0; h < output_blob_correct.dimension(2); h++) {
                for(nn::Index w = 0; w < output_blob_correct.dimension(3); w++) {

                    float correct_value = output_blob_correct(n,c,h,w);
                    float computed_value =  output_blob(n,c,h,w);
                    const float error = correct_value - computed_value;
                    const float squared_error = error*error;

                    std::cout << " expected:" << correct_value << " but computed:" << computed_value << "\n"; 
                    
                    if( squared_error > tolerance ){
                        std::cout << " expected:" << correct_value << " but computed:" << computed_value << "\n"; 
                        return -1;
                    }
                    
                }
            }
        }
    }
    
    return 0;
    
}
