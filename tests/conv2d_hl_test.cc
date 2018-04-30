#include <H5Cpp.h>

#include <iostream>
#include <stdexcept>

#include "tensor.hh"
#include "convolution.hh"
#include "layers.hh"
#include "net.hh"

const double tolerance = 1e-6;

int main(int argc, char* argv[]){

    if(argc != 2){
        std::cerr << "usage: " << argv[0] << " <input h5file>\n";
        return -1;
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

    std::unique_ptr<float[]> input_data(new float[input_size]);
    std::unique_ptr<float[]> kernel_data(new float[kernel_size]);
    
    std::unique_ptr<float[]> output_data_correct(new float[output_size]);

    // load the input and correct result
    input.read(input_data.get(), H5::PredType::NATIVE_FLOAT);
    kernel.read(kernel_data.get(), H5::PredType::NATIVE_FLOAT);

    uint64_t stride_;
    stride.read(&stride_, H5::PredType::NATIVE_UINT64);
    
    uint64_t zero_padding_;
    zero_padding.read(&zero_padding_, H5::PredType::NATIVE_UINT64);
    output.read(output_data_correct.get(), H5::PredType::NATIVE_FLOAT);

    nn::Tensor<float, 4> input_blob{input_data.get(), input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    nn::Tensor<float, 4> kernel_blob{kernel_data.get(), kernel_dims[0], kernel_dims[1], kernel_dims[2], kernel_dims[3]};

    nn::Tensor<float, 4> output_blob_correct{output_data_correct.get(), output_dims[0], output_dims[1], output_dims[2], output_dims[3]};

    nn::Net simple_cnn{};
    simple_cnn += nn::make_convolution_from_hdf5(input_dims[0],
                                                 kernel_dims[0],
                                                 ((input_dims[2] - kernel_dims[2] + 2*zero_padding_) / stride_) + 1,
                                                 ((input_dims[3] - kernel_dims[3] + 2*zero_padding_) / stride_) + 1,
                                                 test_file,
                                                 "kernel", stride_, zero_padding_);
    
    // run the network
    auto output_blob = simple_cnn.forward(input_blob);
    
    // check output blob
    for(nn::Index n = 0; n < output_blob_correct.dimension(0); n++) {
        for(nn::Index c = 0; c < output_blob_correct.dimension(1); c++) {
            for(nn::Index h = 0; h < output_blob_correct.dimension(2); h++) {
                for(nn::Index w = 0; w < output_blob_correct.dimension(3); w++) {

                    float error = output_blob(n,c,h,w) - output_blob_correct(n,c,h,w);
                    float sq_error = error * error;

                    if(sq_error > tolerance){

                        std::cout << "stride:" << stride_ << " padding:" << zero_padding_ << std::endl;

                        std::cout << __FILE__ << ". There was an error in the computed value" << std::endl;
                        std::cout << __FILE__ << ". Expected:" << output_blob_correct(n,c,h,w) << " computed:" << output_blob(n,c,h,w) << std::endl;
                        return -1;
                    }

                }
            }
        }
    }

    std::cout << "success! (no error)" << std::endl;    
    
    return 0;
    
}
