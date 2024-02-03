#include "cuda/common.cuh"
#include "cuda/encryption.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cstdint>

bool test_three_b_adapter(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length, const unsigned int number_threads) {
	const auto raw_size = length * sizeof(std::uint64_t);

	std::uint64_t* values_dev = nullptr;
	std::uint64_t* hashes_dev = nullptr;

	cudaMalloc((void**)&values_dev, raw_size);
	cudaMalloc((void**)&hashes_dev, raw_size);

	cudaMemcpy(values_dev, values, raw_size, cudaMemcpyHostToDevice);

	auto thread_dim = dim3{ number_threads, 1, 1 };
	auto block_dim = dim3{ 1, 1, 1 };

	flat_hash << <block_dim, thread_dim, HASH_SHARED_MEM >> > (values_dev, hashes_dev, length);

	cudaMemcpy(hashes, hashes_dev, raw_size, cudaMemcpyDeviceToHost);

	cudaFree(values_dev);
	cudaFree(hashes_dev);

	cudaDeviceSynchronize();

	auto last_error = cudaGetLastError();

	return last_error == cudaSuccess;
}
