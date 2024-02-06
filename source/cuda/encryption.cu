#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length){
    int x_global = blockIdx.x * blockDim.x + threadIdx.x;
    if(x_global < length){
        constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
        constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
        constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

        const auto val_1 = (value >> 14) + val_a;
        const auto val_2 = (value << 54) ^ val_b;
        const auto val_3 = (val_1 + val_2) << 4;
        const auto val_4 = (value % val_c) * 137;

        hashes[x_global] = val_3 ^ val_4;
        //return final_hash;
    }

__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length){
    int x_global = threadIdx.x;
    if(x_global < length){
        constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
        constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
        constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

        const auto val_1 = (value >> 14) + val_a;
        const auto val_2 = (value << 54) ^ val_b;
        const auto val_3 = (val_1 + val_2) << 4;
        const auto val_4 = (value % val_c) * 137;

        hashes[x_global] = val_3 ^ val_4;
        //return final_hash;
    }

}
