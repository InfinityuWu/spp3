#include "image.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "common.cuh"

// Task 2b)
__global__ void grayscale_kernel (const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height) {
  int x_global = blockIdx.x * blockDim.x + threadIdx.x;
  int y_global = blockIdx.y * blockDim.y + threadIdx.y;

  if (y_global < height && x_global < width) {
    const auto pixel = input[y_global * width + x_global];
    const auto r = pixel.get_red_channel();
    const auto g = pixel.get_green_channel();
    const auto b = pixel.get_blue_channel();
  
    const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
    const auto gray_converted = static_cast<std::uint8_t>(gray);
  
    const auto gray_pixel = BitmapPixel{ gray_converted , gray_converted,  gray_converted };
  
    output[y_global * width + x_global] = gray_pixel;
  }
}

// Task 2c)
BitmapImage get_grayscale_cuda (const BitmapImage& source) {
  auto output_image = BitmapImage{source.get_height(), source.get_width()};
  int number_threads_per_block = 16;
  Pixel<std::uint8_t>** input_gpu;
  Pixel<std::uint8_t>** output_gpu;
  cudaMalloc((void**) &input_gpu, source.get_height() * source.get_width() * sizeof(Pixel<std::uint8_t>));
  cudaMalloc((void**) &output_gpu, source.get_height() * source.get_width() * sizeof(Pixel<std::uint8_t>));
  cudaMemcpy(input_gpu, source.get_data(), source.get_height() * source.get_width() * sizeof(Pixel<std::uint8_t>), cudaMemcpyHostToDevice);
  grayscale_kernel<<< {divup(source.get_height(), number_threads_per_block), divup(source.get_width(), number_threads_per_block)}, {number_threads_per_block, number_threads_per_block} >>>(input_gpu, output_gpu, source.get_width(), source.get_height());
  cudaMemcpy(output_image.get_data(), output_gpu, source.get_height() * source.get_width() * sizeof(Pixel<std::uint8_t>), cudaMemcpyDeviceToHost);
  return output_image;
}
