#include "convolution.hh"
#include "tensor.hh"

#include <iostream>

void nn::conv2d(const Tensor<float, 4> input, const Tensor<float, 4> kernel,
                Tensor<float, 4> output, const size_t stride,
                const size_t zero_padding) {
  assert(input.dimension(0) == output.dimension(0));
  assert(output.dimension(1) == kernel.dimension(0));
  assert(input.dimension(1) == kernel.dimension(1));

  assert(h_extent ==
         (input.dimension(2) - kernel.dimension(2) + 2 * zero_padding) /
                 stride +
             1);
  assert(w_extent ==
         (input.dimension(3) - kernel.dimension(3) + 2 * zero_padding) /
                 stride +
             1);

  for (nn::Index i = 0; i < input.dimension(0); i++) {
    for (nn::Index j = 0; j < output.dimension(1); j++) {
      for (nn::Index n = 0; n < output.dimension(2); n++) {
        for (nn::Index m = 0; m < output.dimension(3); m++) {
          float val = 0.0;

          const int64_t y = static_cast<int64_t>(stride * n) - zero_padding;
          const int64_t x = static_cast<int64_t>(stride * m) - zero_padding;

          for (nn::Index k = 0; k < input.dimension(1); k++) {
            for (nn::Index h = 0; h < kernel.dimension(2); h++) {
              for (nn::Index w = 0; w < kernel.dimension(3); w++) {
                const int64_t y_image = y + h;
                const int64_t x_image = x + w;

                if (0 <= y_image and
                    y_image < static_cast<int64_t>(input.dimension(2)) and
                    0 <= x_image and
                    x_image < static_cast<int64_t>(input.dimension(3))) {
                  float kernel_weight = kernel(j, k, h, w);
                  float inp = input(i, k, y_image, x_image);
                  val += kernel_weight * inp;
                }
              }
            }
          }

          output(i, j, n, m) = val;
        }
      }
    }
  }
}
