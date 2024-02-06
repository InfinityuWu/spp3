#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {
    __shared__ constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
    __shared__ constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
    __shared__ constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };
    int x_global = blockIdx.x * blockDim.x + threadIdx.x;
    if(x_global < length){
        std::uint64_t value = values[x_global];
        const auto val_1 = (value >> 14) + val_a;
        const auto val_2 = (value << 54) ^ val_b;
        const auto val_3 = (val_1 + val_2) << 4;
        const auto val_4 = (value % val_c) * 137;

        hashes[x_global] = val_3 ^ val_4;
        //return final_hash;
    }
}

__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length){
    __shared__ constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
    __shared__ constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
    __shared__ constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };
    int x_global = blockIdx.x * blockDim.x + threadIdx.x;

    //New Added for loop to account for case length > blockDim.x
    for(int i = x_global; i < length; i+=blockDim.x){
        if(x_global < length){
            std::uint64_t value = values[x_global];
            const auto val_1 = (value >> 14) + val_a;
            const auto val_2 = (value << 54) ^ val_b;
            const auto val_3 = (val_1 + val_2) << 4;
            const auto val_4 = (value % val_c) * 137;

            hashes[i] = val_3 ^ val_4;
        }
    }
}
__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr){
    int x_global = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_global < length && searched_hash == hashes[x_global]) {

        // We are sorry, that you have to read this ;D

        // First the value in pointer is incremented by one
        // The current index is the new pointer value - 1,
        // since the pointer value is set to 1 by the first accessing thread (and it should start at 0)

        // Finally the indices value is updated at the correct position!! Yeaa!
        indices[atomicAdd(*ptr, 1)-1] = x_global;
    }
}

__global__ void hash_schemes (std::uint64_t* const hashes, const unsigned int length) {
    __shared__ constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
    __shared__ constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
    __shared__ constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };
    int x_global = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_global < length) {
        std::uint64_t value = x_global;
        value += value << 32;
        const auto val_1 = (value >> 14) + val_a;
        const auto val_2 = (value << 54) ^ val_b;
        const auto val_3 = (val_1 + val_2) << 4;
        const auto val_4 = (value % val_c) * 137;
        hashes[x_global] = val_3 ^ val_4;
    }
}

