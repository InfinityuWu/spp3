#include "test_declarations.h"

#include "cuda/common.cuh"
#include "cuda/image.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cstdint>

bool test_two_b_adapter(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height) {	
	const auto raw_size = height * width * sizeof(Pixel<std::uint8_t>);

	Pixel<std::uint8_t>* source_pixel_dev = nullptr;
	Pixel<std::uint8_t>* destination_pixel_dev = nullptr;

	cudaMalloc((void**)&source_pixel_dev, raw_size);
	cudaMalloc((void**)&destination_pixel_dev, raw_size);

	cudaMemcpy(source_pixel_dev, input, raw_size, cudaMemcpyHostToDevice);

	auto thread_dim = dim3{ 32, 32, 1 };
	auto block_dim = dim3{ divup(width, 32), divup(height, 32), 1 };

	grayscale_kernel << <block_dim, thread_dim, GRAYSCALE_SHARED_MEM >> > (source_pixel_dev, destination_pixel_dev, width, height);

	cudaMemcpy(output, destination_pixel_dev, raw_size, cudaMemcpyDeviceToHost);

	cudaFree(source_pixel_dev);
	cudaFree(destination_pixel_dev);

	cudaDeviceSynchronize();

	auto last_error = cudaGetLastError();

	return last_error == cudaSuccess;
}

