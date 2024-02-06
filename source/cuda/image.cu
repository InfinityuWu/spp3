#include "image.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Task 2b)
__global__ void grayscale_kernel(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height) {
  int x_global = blockIdx.x * blockDim.x + threadIdx.x;
  int y_global = blockIdx.y * blockDim.y + threadIdx.y;

  if (y_global < height && x_global < width) {
    const auto pixel = input[y_global * height + x_global];
    const auto r = pixel.get_red_channel();
    const auto g = pixel.get_green_channel();
    const auto b = pixel.get_blue_channel();
  
    const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
    const auto gray_converted = static_cast<std::uint8_t>(gray);
  
    const auto gray_pixel = BitmapPixel{ gray_converted , gray_converted,  gray_converted };
  
    output[y_global * height + x_global] = gray_pixel;
  }
}

BitmapImage get_grayscale_cuda(const BitmapImage& source);
