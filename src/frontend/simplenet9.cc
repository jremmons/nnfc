/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include <turbojpeg.h>

#include <chrono>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include <streambuf>
#include <fstream>

#include "tensor.hh"
#include "layers.hh"
#include "net.hh"

// assumptions
static_assert(sizeof(uint8_t) == sizeof(char), "sizeof(uint8_t) != sizeof(char)");
static_assert(sizeof(uint8_t) == sizeof(unsigned char), "sizeof(uint8_t) != sizeof(unsigned char)");

const int width = 32;
const int height = 32;
const int channels = 3;

const char* labels[] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};  

std::vector<uint8_t> read_image(std::string filename)
{
    std::ifstream file(filename, std::ios::in|std::ios::binary);    
    file.seekg(0, std::ios::end);
    size_t compressed_size = file.tellg();
    file.seekg(0);

    
    std::vector<uint8_t> compressed_buffer(compressed_size);
    file.read(reinterpret_cast<char*>(compressed_buffer.data()), compressed_size);
       
    std::vector<uint8_t> image_buffer(width * height * channels);

    tjhandle jpeg_decompress_context = tjInitDecompress();

    int input_width, input_height, input_subsamp;
    tjDecompressHeader2(jpeg_decompress_context, reinterpret_cast<unsigned char*>(compressed_buffer.data()), compressed_size, &input_width, &input_height, &input_subsamp);

    std::cout << "jpg_width: " << input_width << "\n";
    std::cout << "jpg_height: " << input_height << "\n";
    std::cout << "\n";
    assert(width == input_width);
    assert(height == input_height);
    
    tjDecompress2(jpeg_decompress_context, reinterpret_cast<unsigned char*>(compressed_buffer.data()), compressed_size,
                  reinterpret_cast<unsigned char*>(image_buffer.data()), width, 0/*pitch*/, height, TJPF_RGB, TJFLAG_FASTDCT);

    tjDestroy(jpeg_decompress_context);
    
    return image_buffer;
}


nn::Tensor<float, 4> rgb2tensor(std::vector<uint8_t> image)
{
    nn::Tensor<float, 4> tensor(1, channels, height, width);

    float means[3] = {0.4914, 0.4822, 0.4465};
    float variances[3] = {0.2023, 0.1994, 0.2010};
    
    // copy data into tensor and perform image augmentation
    for(nn::Index c = 0; c < channels; c++){
        for(nn::Index h = 0; h < height; h++){
            for(nn::Index w = 0; w < width; w++){

                nn::Index offset = channels*width*h + channels*w + c;
                uint8_t pixel = image[offset];
                
                tensor(0, c, h, w) = ((static_cast<float>(pixel) / 255) - means[c]) / variances[c];
            }
        }
    }
    
    return tensor;
}

void build_simplenet(H5::H5File &parameter_file, nn::Net &net)
{


    // layer 0
    net += nn::make_convolution_from_hdf5(1, 64, 32, 32,
                                          parameter_file,
                                          "conv0.weight", 1, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 64, 32, 32,
                                         parameter_file,
                                         "bn0.running_mean",
                                         "bn0.running_var",
                                         "bn0.weight",
                                         "bn0.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 64, 32, 32);

    // layer 1
    net += nn::make_convolution_from_hdf5(1, 64, 32, 32,
                                          parameter_file,
                                          "conv1.weight", 1, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 64, 32, 32,
                                         parameter_file,
                                         "bn1.running_mean",
                                         "bn1.running_var",
                                         "bn1.weight",
                                         "bn1.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 64, 32, 32);

    // layer 2
    net += nn::make_convolution_from_hdf5(1, 128, 32, 32,
                                          parameter_file,
                                          "conv2.weight", 1, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 128, 32, 32,
                                         parameter_file,
                                         "bn2.running_mean",
                                         "bn2.running_var",
                                         "bn2.weight",
                                         "bn2.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 128, 32, 32);

    // layer 3 (stride == 2)
    net += nn::make_convolution_from_hdf5(1, 128, 16, 16,
                                          parameter_file,
                                          "conv3.weight", 2, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 128, 16, 16,
                                         parameter_file,
                                         "bn3.running_mean",
                                         "bn3.running_var",
                                         "bn3.weight",
                                         "bn3.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 128, 16, 16);

    // layer 4
    net += nn::make_convolution_from_hdf5(1, 256, 16, 16,
                                          parameter_file,
                                          "conv4.weight", 1, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 256, 16, 16,
                                         parameter_file,
                                         "bn4.running_mean",
                                         "bn4.running_var",
                                         "bn4.weight",
                                         "bn4.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 256, 16, 16);

    // layer 5 (stride == 2)
    net += nn::make_convolution_from_hdf5(1, 256, 8, 8,
                                          parameter_file,
                                          "conv5.weight", 2, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 256, 8, 8,
                                         parameter_file,
                                         "bn5.running_mean",
                                         "bn5.running_var",
                                         "bn5.weight",
                                         "bn5.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 256, 8, 8);
    
    // layer 6
    net += nn::make_convolution_from_hdf5(1, 512, 8, 8,
                                          parameter_file,
                                          "conv6.weight", 1, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 512, 8, 8,
                                         parameter_file,
                                         "bn6.running_mean",
                                         "bn6.running_var",
                                         "bn6.weight",
                                         "bn6.bias",
                                         0.00001);
    
    net += nn::make_relu_from_hdf5(1, 512, 8, 8);

    // layer 7 (stride == 2)
    net += nn::make_convolution_from_hdf5(1, 512, 4, 4,
                                          parameter_file,
                                          "conv7.weight", 2, 1);
    
    net += nn::make_batch_norm_from_hdf5(1, 512, 4, 4,
                                         parameter_file,
                                         "bn7.running_mean",
                                         "bn7.running_var",
                                         "bn7.weight",
                                         "bn7.bias",
                                         0.00001);

    
    net += nn::make_relu_from_hdf5(1, 512, 4, 4);
    
    net += nn::make_pool_from_hdf5(1, 512, 1, 1);

    net += nn::make_fc_with_bias_from_hdf5(1, 10, 1, 1,
                                           parameter_file,
                                           "linear.weight",
                                           "linear.bias");
}

int main(int argc, char* argv[])
{
    
    if(argc != 3) {
        std::cout << "usage: " << argv[0] << " <parameters.h5> <image.jpg>\n";
        return 0;
    }

    // load input image
    std::vector<uint8_t> image_buffer = read_image(argv[2]);
    nn::Tensor<float, 4> image_tensor = rgb2tensor(image_buffer);

    for(nn::Index i = 0; i < 4; i++) {
        std::cout << "image_tensor.dimension(" << i << "): " << image_tensor.dimension(i) << "\n";
    }
    std::cout << "\n";
    
    // build model and load parameters
    H5::H5File parameter_file(argv[1], H5F_ACC_RDONLY);
    nn::Net simple_cnn{};

    build_simplenet(parameter_file, simple_cnn);

    // perform the forward pass
    auto t1 = std::chrono::high_resolution_clock::now();
    nn::Tensor<float, 4> prediction = simple_cnn.forward(image_tensor);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    
    // log the output the console
    float top_val = prediction(0, 0, 0, 0);
    int top_prediction = 0;
    for(nn::Index i = 0; i < 10; i++) {
        float val = prediction(0, i, 0, 0);
        std::cout << "prediction(0, " << i << ", 0, 0): " << val << "\n";

        if(top_val < val){
            top_val = val;
            top_prediction = i;
        }
    }

    std::cout << "\n" << "CNN predicted: " << labels[top_prediction] << ". (score: " << top_val << ")\n";
    std::cout << "The prediction took: " << time_span.count() << " seconds\n";
    
    return 0;
}
